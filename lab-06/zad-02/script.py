import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

data_dir = r'C:\Users\Admin\Downloads\dogs-cats-mini\dogs-cats-mini'

# definicja naszej klasy czytającej dane
class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # wszystkie pliki .jpg z folderu
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # wczytujemy obraz
        image = Image.open(img_path).convert('RGB')
        
        # oznaczamy etykiety: jeśli w nazwie pliku jest 'cat', to etykieta=0. w przeciwnym razie 1 (pies).
        label = 0 if 'cat' in img_name.lower() else 1
        
        # przekształcamy obraz (np. zmiana rozmiaru, zamiana na tensor)
        if self.transform:
            image = self.transform(image)
            
        return image, label

# transformacje (ujednolicamy rozmiar do 128x128 i zamieniamy na format zrozumiały dla AI - Tensor)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# tworzymy główny zbiór danych
full_dataset = CatsDogsDataset(root_dir=data_dir, transform=transform)

# dzielimy zbiór na treningowy (80%) i walidacyjny (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# pakujemy dane do "Ładowarek" (DataLoaders), które będą karmić model paczkami po 32 zdjęcia
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# test: sprawdźmy, czy działa! wyświetlimy pierwsze wczytane zdjęcie.
images, labels = next(iter(train_loader))
plt.imshow(images[0].permute(1, 2, 0)) # Permute jest potrzebne, bo PyTorch inaczej układa kolory niż matplotlib
plt.title(f"Etykieta: {'Kot' if labels[0].item() == 0 else 'Pies'}")
plt.axis('off')
plt.show()

print(f"dane wczytane! zbiór treningowy: {len(train_dataset)} zdjęć. zbiór walidacyjny: {len(val_dataset)} zdjęć.")

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # warstwy konwolucyjne (wykrywanie cech)
        self.conv_layers = nn.Sequential(
            # wejście: 3 kanały (RGB), 16 filtrów, rozmiar filtra 3x3
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # zmniejsza obraz ze 128x128 na 64x64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # zmniejsza na 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # zmniejsza na 16x16
        )
        
        # warstwy w pełni połączone (podejmowanie decyzji)
        self.fc_layers = nn.Sequential(
            nn.Flatten(), # zamienia obraz 16x16x64 na jeden długi wektor
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # losowo wyłącza neurony, żeby sieć nie "wkuwała" zdjęć na pamięć
            nn.Linear(512, 2) # wyjście: 2 klasy (Kot lub Pies)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# # tworzymy model i wysyłamy go na GPU, jeśli jest dostępne
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

print(f"Model został zbudowany i wysłany na: {device}")
print(model)

import torch.optim as optim
import time

# # 0. zarządzanie checkpointami
model_path = 'cat_dog_model.pth'
criterion = nn.CrossEntropyLoss()

# if os.path.exists(model_path):
#     # Wczytujemy wagi do zainicjalizowanego modelu
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     print("Wczytano gotowy model z pliku!")
#     train_losses, val_losses = [], []
#     train_accs, val_accs = [], []

#     # Liczymy metryki walidacyjne dla wczytanego modelu, aby móc go porównać z ResNet.
#     model.eval()
#     loaded_val_loss, loaded_val_correct, loaded_val_total = 0.0, 0, 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loaded_val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             loaded_val_total += labels.size(0)
#             loaded_val_correct += (predicted == labels).sum().item()

#     val_losses.append(loaded_val_loss / len(val_loader))
#     val_accs.append(loaded_val_correct / loaded_val_total)
# else:
print("Brak zapisanego modelu, rozpoczynam trening od zera...")

# # 1. ustawienia treningu
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 3

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Rozpoczynam trening...")

for epoch in range(epochs):
    start_time = time.time()
    
    # faza treningu
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)
    
    # faza walidacji
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_correct / val_total)
    
    end_time = time.time()
    print(f"Epoka {epoch+1}/{epochs} ({end_time - start_time:.1f}s) -> "
            f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_accs[-1]:.4f} | "
            f"Val Loss: {val_losses[-1]:.4f}, Acc: {val_accs[-1]:.4f}")

# # Zapisujemy wyuczone wagi do pliku .pth
torch.save(model.state_dict(), model_path)
print(f"Model został zapisany do pliku {model_path}")

# # 2. rysowanie wykresów (krzywe uczenia)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Strata Treningowa')
plt.plot(val_losses, label='Strata Walidacyjna')
plt.title('Funkcja Straty (Loss)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Dokładność Treningowa')
plt.plot(val_accs, label='Dokładność Walidacyjna')
plt.title('Dokładność (Accuracy)')
plt.legend()
plt.show()

from torchvision import models

# pobieramy gotowy model ResNet18
resnet = models.resnet18(weights='IMAGENET1K_V1')

# "zamrażamy" go, żeby nie uczył się od zera wszystkiego
for param in resnet.parameters():
    param.requires_grad = False

# podmieniamy tylko ostatnią warstwę na naszą (2 wyjścia: pies/kot)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 2)

resnet = resnet.to(device)

# ustawiamy trening tylko dla tej ostatniej warstwy
optimizer_resnet = optim.Adam(resnet.fc.parameters(), lr=0.001)

print("Model ResNet gotowy do szybkiego dotrenowania!")

# # 1. Ustawienia treningu dla ResNet (Transfer Learning)
# (criterion zostaje ten sam, bo to nadal klasyfikacja)
resnet_epochs = 3 
resnet_model_path = 'cat_dog_resnet_model.pth'

resnet_train_losses, resnet_val_losses = [], []
resnet_train_accs, resnet_val_accs = [], []

# if os.path.exists(resnet_model_path):
#     resnet.load_state_dict(torch.load(resnet_model_path, map_location=device))
#     resnet.to(device)
#     print("\nWczytano gotowy model ResNet z pliku!")

#     resnet.eval()
#     loaded_resnet_val_loss, loaded_resnet_val_correct, loaded_resnet_val_total = 0.0, 0, 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = resnet(images)
#             loss = criterion(outputs, labels)
#             loaded_resnet_val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             loaded_resnet_val_total += labels.size(0)
#             loaded_resnet_val_correct += (predicted == labels).sum().item()

#     resnet_val_losses.append(loaded_resnet_val_loss / len(val_loader))
#     resnet_val_accs.append(loaded_resnet_val_correct / loaded_resnet_val_total)
# else:
print("\nRozpoczynam trening ResNet (Transfer Learning)...")

for epoch in range(resnet_epochs):
    start_time = time.time()
    
    # FAZA TRENINGU - używamy 'resnet'
    resnet.train() 
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer_resnet.zero_grad() # ZMIANA na optimizer_resnet
        outputs = resnet(images)      # ZMIANA na resnet
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_resnet.step()      # ZMIANA na optimizer_resnet
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    resnet_train_losses.append(running_loss / len(train_loader))
    resnet_train_accs.append(correct / total)
    
    # FAZA WALIDACJI - używamy 'resnet'
    resnet.eval() 
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images) # ZMIANA na resnet
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    resnet_val_losses.append(val_loss / len(val_loader))
    resnet_val_accs.append(val_correct / val_total)
    
    end_time = time.time()
    print(f"Epoka {epoch+1}/{resnet_epochs} ({end_time - start_time:.1f}s) -> "
            f"ResNet Train Loss: {resnet_train_losses[-1]:.4f}, Acc: {resnet_train_accs[-1]:.4f} | "
            f"ResNet Val Acc: {resnet_val_accs[-1]:.4f}")

torch.save(resnet.state_dict(), resnet_model_path)
print(f"Model ResNet został zapisany do pliku {resnet_model_path}")

# # 2. Rysowanie wykresów porównawczych
print("\n=== Porównanie: Model od zera vs ResNet Transfer Learning ===")
print(f"Model od zera - końcowa dokładność walidacyjna: {val_accs[-1]:.4f}")
print(f"ResNet TL - końcowa dokładność walidacyjna: {resnet_val_accs[-1]:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o', label='Model od zera (Train)')
plt.plot(resnet_train_losses, marker='s', label='ResNet TL (Train)', linestyle='--')
plt.title('Porównanie Loss - Faza Treningowa')
plt.xlabel('Epoka')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(val_accs, marker='o', label='Model od zera (Val)')
plt.plot(resnet_val_accs, marker='s', label='ResNet TL (Val)', linestyle='--')
plt.title('Porównanie Dokładności - Faza Walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

import numpy as np

model.eval()
pomyłki_obrazy = []
pomyłki_prawda = []
pomyłki_strzał = []

# musimy odwrócić normalizację, żeby obrazki na ekranie miały ładne kolory
def imshow_tensor(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    return img

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # znajdź indeksy, gdzie model się pomylił
        maska_bledow = preds != labels
        if maska_bledow.any():
            bledne_obrazy = images[maska_bledow]
            prawdziwe_etykiety = labels[maska_bledow]
            przewidziane_etykiety = preds[maska_bledow]
            
            for i in range(len(bledne_obrazy)):
                if len(pomyłki_obrazy) < 5:
                    pomyłki_obrazy.append(bledne_obrazy[i])
                    pomyłki_prawda.append(prawdziwe_etykiety[i].item())
                    pomyłki_strzał.append(przewidziane_etykiety[i].item())
                else:
                    break
        if len(pomyłki_obrazy) >= 5:
            break

# wyświetlanie pomyłek
nazwy_klas = ['Kot', 'Pies']
plt.figure(figsize=(15, 5))
for i in range(len(pomyłki_obrazy)):
    plt.subplot(1, 5, i+1)
    plt.imshow(imshow_tensor(pomyłki_obrazy[i]))
    plt.title(f"Prawda: {nazwy_klas[pomyłki_prawda[i]]}\nModel: {nazwy_klas[pomyłki_strzał[i]]}")
    plt.axis('off')
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Kot', 'Pies'], yticklabels=['Kot', 'Pies'], cmap='Blues')
plt.xlabel('Przewidziane')
plt.ylabel('Prawdziwe')
plt.title('Macierz Pomyłek')
plt.show()