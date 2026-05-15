"""
Trening klasyfikatora emocji MobileNetV3 (Large) z transfer learning na RAF-DB.

RAF-DB: 7 klas emocji (1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness,
6=Anger, 7=Neutral), obrazy aligned 100x100 RGB, ~12k train, ~3k test.

Strategia transfer learning (dwie fazy):
- Faza 1 (warm-up): backbone zamrozony, trenujemy tylko nowa glowe klasyfikatora.
  Wyzszy LR (1e-3), kilka epok zeby glowa "zlapala" sygnal.
- Faza 2 (fine-tuning): odmrazamy caly model, niski LR (1e-4) z cosine annealing,
  wagi ImageNet sa delikatnie dostrajane do dziedziny twarzy.

Funkcje:
- Pretrained MobileNetV3-Large (ImageNet) z torchvision
- Augmentacja (resize 224, flip, rotation, color jitter, random erasing)
- Normalizacja statystykami ImageNet
- Wagi klas (kompensacja imbalance RAF-DB: Happiness ~5k vs Disgust ~700)
- LR scheduler (CosineAnnealingLR w fazie fine-tuningu)
- Early stopping na val accuracy
- Krzywe uczenia, confusion matrix, classification report
- Zapis najlepszego modelu wg val accuracy

Uruchomienie lokalnie:
    python train_mobilenetv3.py --data_dir ./data/RAF-DB/DATASET --epochs 20

Na Google Colab (patrz notebooks/train_mobilenetv3.ipynb):
    !python train_mobilenetv3.py --data_dir /content/RAF-DB/DATASET \
        --output_dir /content/out --epochs 20
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.datasets import ImageFolder


RAF_DB_CLASS_NAMES = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data/RAF-DB/DATASET",
                   help="Folder z podkatalogami train/ i test/ (klasy 1..7)")
    p.add_argument("--output_dir", default="./experiments/mobilenetv3",
                   help="Folder na wagi, wykresy i metryki")
    p.add_argument("--epochs", type=int, default=20,
                   help="Laczna liczba epok (warmup_epochs + fine-tuning)")
    p.add_argument("--warmup_epochs", type=int, default=3,
                   help="Epoki z zamrozonym backbone (trening samej glowy)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_head", type=float, default=1e-3,
                   help="LR w fazie warm-up (tylko glowa)")
    p.add_argument("--lr_finetune", type=float, default=1e-4,
                   help="LR w fazie fine-tuningu (caly model)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience (epoki bez poprawy val acc)")
    p.add_argument("--img_size", type=int, default=224,
                   help="Rozmiar wejsciowy modelu (MobileNetV3 oczekuje 224)")
    p.add_argument("--variant", choices=["large", "small"], default="large",
                   help="Wariant MobileNetV3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_augment", action="store_true",
                   help="Wylacza augmentacje (do eksperymentu porownawczego)")
    return p.parse_args()


def build_transforms(img_size: int, use_augmentation: bool):
    # Statystyki ImageNet -- pretrained backbone byl trenowany na tej normalizacji
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if not use_augmentation:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    return train_tf, eval_tf


def split_dataset(train_dir, train_tf, eval_tf, val_split, seed):
    train_full = ImageFolder(root=train_dir, transform=train_tf)
    val_full = ImageFolder(root=train_dir, transform=eval_tf)
    n = len(train_full)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split = int((1 - val_split) * n)
    train_idx = indices[:split].tolist()
    val_idx = indices[split:].tolist()
    train_targets = [train_full.targets[i] for i in train_idx]
    return (
        Subset(train_full, train_idx),
        Subset(val_full, val_idx),
        train_full.classes,
        train_targets,
    )


def compute_class_weights(targets, num_classes, device):
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    # standardowa formula sklearn 'balanced': N / (K * n_k)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_mobilenetv3(num_classes: int, variant: str = "large"):
    """Zwraca MobileNetV3 z pretrained ImageNet i podmieniona glowa klasyfikatora.

    Klasyfikator MobileNetV3 to Sequential[Linear(in, hidden), Hardswish,
    Dropout, Linear(hidden, num_classes)]. Zachowujemy hidden + dropout,
    podmieniamy tylko ostatni Linear -- standardowy patent transfer learning.
    """
    if variant == "large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
    else:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train(train_mode)
    total_loss, total_correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    return total_loss / total, 100.0 * total_correct / total


def evaluate_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def save_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="val", marker="s")
    axes[0].axvline(x=history["warmup_end"] + 0.5, color="red",
                    linestyle="--", alpha=0.5, label="koniec warm-up")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], label="train", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="val", marker="s")
    axes[1].axvline(x=history["warmup_end"] + 0.5, color="red",
                    linestyle="--", alpha=0.5, label="koniec warm-up")
    axes[1].set_title("Accuracy [%]")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawda")
    ax.set_title("Confusion matrix (test, znormalizowana)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def label_classes(folder_classes):
    """Mapuje nazwy folderow ('1'..'7') na czytelne nazwy emocji."""
    return [RAF_DB_CLASS_NAMES.get(c, c) for c in folder_classes]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size, use_augmentation=not args.no_augment)
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    train_set, val_set, folder_classes, train_targets = split_dataset(
        train_dir, train_tf, eval_tf, args.val_split, args.seed
    )
    test_set = ImageFolder(root=test_dir, transform=eval_tf)
    classes = label_classes(folder_classes)

    print(f"Klasy (folder -> nazwa): {dict(zip(folder_classes, classes))}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"Augmentacja: {'TAK' if not args.no_augment else 'NIE'}")
    print(f"Model: MobileNetV3-{args.variant} (pretrained ImageNet)")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Urzadzenie: {device}")

    model = build_mobilenetv3(num_classes=len(classes), variant=args.variant).to(device)
    class_weights = compute_class_weights(train_targets, len(classes), device)
    print(f"Wagi klas: {dict(zip(classes, class_weights.cpu().numpy().round(3).tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": [],
        "warmup_end": args.warmup_epochs,
    }
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_path = out_dir / "mobilenetv3_best.pth"

    # === FAZA 1: warm-up (zamrozony backbone, trening tylko glowy) ===
    print(f"\n=== Faza 1: warm-up ({args.warmup_epochs} epok, zamrozony backbone) ===")
    freeze_backbone(model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=args.weight_decay,
    )
    scheduler = None

    for epoch in range(1, args.epochs + 1):
        # Po fazie warm-up odmrazamy caly model i przelaczamy optimizer/scheduler
        if epoch == args.warmup_epochs + 1:
            print(f"\n=== Faza 2: fine-tuning (odmrozony caly model, LR={args.lr_finetune}) ===")
            unfreeze_all(model)
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr_finetune,
                weight_decay=args.weight_decay,
            )
            remaining_epochs = max(1, args.epochs - args.warmup_epochs)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_epochs, eta_min=1e-6
            )

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train_mode=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train_mode=False
        )
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        phase = "warmup" if epoch <= args.warmup_epochs else "finetune"
        print(
            f"Epoka [{epoch:02d}/{args.epochs}] ({phase}) | "
            f"Train loss {train_loss:.4f} acc {train_acc:5.2f}% | "
            f"Val loss {val_loss:.4f} acc {val_acc:5.2f}% | "
            f"LR {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> nowy best (val acc {val_acc:.2f}%) zapisany")
        else:
            epochs_no_improve += 1
            # Early stopping wlacza sie dopiero w fazie fine-tuningu
            if epoch > args.warmup_epochs and epochs_no_improve >= args.patience:
                print(f"Early stopping po {epoch} epokach (brak poprawy przez {args.patience})")
                break

    save_curves(history, out_dir / "learning_curves.png")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nKrzywe uczenia: {out_dir / 'learning_curves.png'}")

    print("\n=== Ewaluacja na zbiorze testowym (najlepszy model) ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred = evaluate_predictions(model, test_loader, device)
    test_acc = 100.0 * (y_true == y_pred).mean()
    print(f"Test accuracy: {test_acc:.2f}%\n")

    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Model: MobileNetV3-{args.variant} (transfer learning, ImageNet -> RAF-DB)\n")
        f.write(f"Test accuracy: {test_acc:.2f}%\n")
        f.write(f"Best val accuracy: {best_val_acc:.2f}%\n\n")
        f.write(report)

    save_confusion_matrix(y_true, y_pred, classes, out_dir / "confusion_matrix.png")
    print(f"Confusion matrix: {out_dir / 'confusion_matrix.png'}")

    # Zapis konfiguracji eksperymentu (do sprawozdania)
    config = vars(args).copy()
    config["test_accuracy"] = float(test_acc)
    config["best_val_accuracy"] = float(best_val_acc)
    config["classes"] = classes
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nGotowe. Wszystkie artefakty w: {out_dir}/")


if __name__ == "__main__":
    main()
