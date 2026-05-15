import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import EmotionCNN
import torch.nn as nn
import torch.optim as optim

# Ustawiamy transformacje dla zbioru treningowego
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Wymuszamy 1 kanał (czarno-białe)
    transforms.ToTensor(),                       # Zamieniamy obrazek na Tensor PyTorcha
    transforms.Normalize((0.5,), (0.5,))         # Normalizujemy wartości pikseli z [0, 1] na [-1, 1]
])

# Ścieżka do folderu z danymi treningowymi
data_dir = './FER2013/train' 

# Wczytujemy dataset
train_dataset = ImageFolder(root=data_dir, transform=transform)

# Tworzymy DataLoader - on będzie "karmił" naszą sieć paczkami (batch) po 64 obrazki na raz
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Sprawdzamy, jakie klasy (emocje) wykrył PyTorch na podstawie nazw folderów
classes = train_dataset.classes
print(f"Znalezione klasy emocji: {classes}")
print(f"Liczba obrazów treningowych: {len(train_dataset)}")

def imshow(img):
    img = img / 2 + 0.5  # Odkręcamy normalizację (z [-1, 1] wracamy na [0, 1])
    npimg = img.numpy()
    # PyTorch trzyma obrazy w formacie [Kanały, Wysokość, Szerokość]
    # Matplotlib wymaga formatu [Wysokość, Szerokość, Kanały]
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# Pobieramy jedną paczkę (batch) danych treningowych
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Wyświetlamy 8 pierwszych obrazków z paczki
print(" | ".join(f"{classes[labels[j]]:5s}" for j in range(8)))
imshow(torchvision.utils.make_grid(images[:8]))
    
# Inicjalizujemy model. Zbiór FER-2013 zazwyczaj ma 7 klas emocji
num_classes = len(classes) # Zmienna 'classes' pochodzi z poprzedniego kodu
model = EmotionCNN(num_classes=num_classes)

print(model)

# Test z "pustym" obrazkiem o odpowiednim wymiarze: [Rozmiar_paczki, Kanały, Wysokość, Szerokość]
# Dajemy 1 obrazek (batch_size=1), 1 kanał (czarno-biały), 48x48 pikseli
dummy_input = torch.randn(1, 1, 48, 48)
output = model(dummy_input)

print(f"\nKształt na wejściu: {dummy_input.shape}")
print(f"Kształt na wyjściu: {output.shape} -> [Paczka, Liczba klas]")

# Sprawdzamy, czy mamy dostępną kartę graficzną (CUDA/MPS) czy tylko procesor (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Będziemy trenować na: {device}")

# Przenosimy nasz model na wybrane urządzenie
model = model.to(device)

# --- Ładowanie danych testowych (do sprawdzania wiedzy sieci) ---
# Zakładam, że obok folderu 'train' masz folder 'test'
test_dir = './FER2013/test' 
test_dataset = ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Definiujemy Kryterium i Optymalizator
criterion = nn.CrossEntropyLoss() # Standardowa funkcja błędu dla klasyfikacji
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam to jeden z najlepszych optymalizatorów

# Liczba epok (ile razy sieć zobaczy cały zbiór danych)
epochs = 10

print("Rozpoczynamy trening...")

for epoch in range(epochs):
    model.train() # Ustawiamy model w tryb treningowy (włącza to np. Dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Przechodzimy przez paczki danych z train_loader
    for images, labels in train_loader:
        # Przenosimy dane na kartę graficzną lub procesor
        images, labels = images.to(device), labels.to(device)
        
        # 1. Zerujemy gradienty (czyli "czyścimy tablicę" przed nowymi obliczeniami)
        optimizer.zero_grad()
        
        # 2. Przewidywanie (Forward pass)
        outputs = model(images)
        
        # 3. Obliczanie błędu (Loss)
        loss = criterion(outputs, labels)
        
        # 4. Obliczanie gradientów (Backward pass)
        loss.backward()
        
        # 5. Aktualizacja wag (Krok optymalizatora)
        optimizer.step()
        
        # Statystyki do wyświetlania
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    # Dokładność na danych treningowych
    train_accuracy = 100 * correct / total
    
    # --- FAZA TESTOWANIA (Ewaluacja) ---
    model.eval() # Wyłączamy Dropout i inne mechanizmy uczące na czas testu
    test_correct = 0
    test_total = 0
    
    # Wyłączamy obliczanie gradientów (oszczędza to mnóstwo pamięci i czasu)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    test_accuracy = 100 * test_correct / test_total
    
    print(f"Epoka [{epoch+1}/{epochs}] | Błąd (Loss): {running_loss/len(train_loader):.4f} | Dokładność treningowa: {train_accuracy:.2f}% | Dokładność testowa: {test_accuracy:.2f}%")

print("Zakończono trening")

# Zapisujemy wyuczone wagi do pliku
torch.save(model.state_dict(), 'emotion_model.pth')
print("Model zapisany jako 'emotion_model.pth'")

