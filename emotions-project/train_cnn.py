"""
Trening klasyfikatora emocji EmotionCNN na FER2013.

Funkcje:
- Augmentacja danych (flip, rotation, affine, color jitter, random erasing)
- Split train/val 80/20 ze staym seedem
- Wagi klas (kompensacja imbalance FER2013: Disgust ~500 vs Happy ~7000)
- LR scheduler (ReduceLROnPlateau)
- Early stopping na val accuracy
- Zapis krzywych uczenia (loss/accuracy PNG)
- Confusion matrix + classification report na zbiorze testowym
- Zapis najlepszego modelu na podstawie val accuracy

Uruchomienie lokalnie:
    python train.py --data_dir ./data/FER2013 --epochs 30

Na Google Colab (patrz notebooks/train_cnn.ipynb):
    !python train.py --data_dir /content/FER2013 --output_dir /content/out --epochs 30
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
from torchvision.datasets import ImageFolder

from model import EmotionCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data/FER2013",
                   help="Folder z podkatalogami train/ i test/")
    p.add_argument("--output_dir", default="./experiments/cnn_augmented",
                   help="Folder na wagi, wykresy i metryki")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience (epoki bez poprawy val acc)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_augment", action="store_true",
                   help="Wylacza augmentacje (do eksperymentu porownawczego)")
    return p.parse_args()


def build_transforms(use_augmentation: bool):
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    if not use_augmentation:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
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
    # waga = N / (K * n_k) -- standardowa formula sklearn 'balanced'
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


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
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], label="train", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="val", marker="s")
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(use_augmentation=not args.no_augment)
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    train_set, val_set, classes, train_targets = split_dataset(
        train_dir, train_tf, eval_tf, args.val_split, args.seed
    )
    test_set = ImageFolder(root=test_dir, transform=eval_tf)

    print(f"Klasy: {classes}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"Augmentacja: {'TAK' if not args.no_augment else 'NIE'}")

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

    model = EmotionCNN(num_classes=len(classes)).to(device)
    class_weights = compute_class_weights(train_targets, len(classes), device)
    print(f"Wagi klas: {dict(zip(classes, class_weights.cpu().numpy().round(3).tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_path = out_dir / "emotion_model_best.pth"

    print("\n=== Trening ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train_mode=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train_mode=False
        )
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoka [{epoch:02d}/{args.epochs}] | "
            f"Train loss {train_loss:.4f} acc {train_acc:5.2f}% | "
            f"Val loss {val_loss:.4f} acc {val_acc:5.2f}% | "
            f"LR {current_lr:.5f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> nowy best (val acc {val_acc:.2f}%) zapisany")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
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
        f.write(f"Test accuracy: {test_acc:.2f}%\n\n")
        f.write(report)

    save_confusion_matrix(y_true, y_pred, classes, out_dir / "confusion_matrix.png")
    print(f"Confusion matrix: {out_dir / 'confusion_matrix.png'}")

    # Plik kompatybilny ze starym detect.py / webapp
    torch.save(model.state_dict(), out_dir / "emotion_model.pth")
    print(f"\nGotowe. Wszystkie artefakty w: {out_dir}/")


if __name__ == "__main__":
    main()
