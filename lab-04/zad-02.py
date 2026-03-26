import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# A) Narzędzia do wczytywania i obróbki (DataLoading)

class IrisDataset(Dataset):
    def __init__(self, features, labels):
        # Konwersja do tensorów PyTorch
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# B) Budowa modelu sieci neuronowej

class IrisNet(nn.Module):
    """
    STRUKTURA SIECI (Topologia):
    Dla problemu Iris mamy 4 cechy wejściowe (np. długość i szerokość płatka/działki)
    oraz 3 klasy wyjściowe (Setosa, Versicolor, Virginica).
    
    Ponieważ problem jest relatywnie prosty, używamy prostej sieci MLP 
    (Multilayer Perceptron - perceptron wielowarstwowy):
    - Warstwa wejściowa: 4 neurony (odpowiadające 4 cechom).
    - 1. Warstwa ukryta: 16 neuronów z funkcją aktywacji ReLU. 
        16 to liczba "z zapasem" pozwalająca wychwycić nieliniowe zależności, ale na tyle 
        mała, by nie doprowadzić do szybkiego przeuczenia (overfittingu).
    - 2. Warstwa ukryta: 16 neuronów z funkcją aktywacji ReLU (pogłębia możliwości decyzyjne sieci).
    - Warstwa wyjściowa: 3 neurony (dla każdej z klas)
    """
    def __init__(self):
        super(IrisNet, self).__init__()  
        self.fc1 = nn.Linear(4, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def main():
    try:
        df = pd.read_csv('../iris_big.csv', header=0)
    except FileNotFoundError:
        print("Nie znaleziono pliku 'iris_big.csv'")
        return

    X_raw = df.iloc[:, :-1].values # Wszystkie kolumny oprócz ostatniej
    y_raw = df.iloc[:, -1].values  # Tylko ostatnia kolumna

    # 2. Obróbka i normalizacja
    # Kodowanie etykiet tekstowych (np. "Iris-setosa") na liczby (0, 1, 2)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_encoded, test_size=0.2, random_state=42)

    # Normalizacja danych
    # Standaryzujemy cechy do średniej 0 i odchylenia standardowego 1.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = IrisDataset(X_train_scaled, y_train)
    val_dataset = IrisDataset(X_val_scaled, y_val)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Inicjalizacja modelu, funkcji straty i optymalizatora
    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    
    # Listy do przechowywania historii uczenia
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # C) Pętla trenująca w PyTorch
    for epoch in range(epochs):
        model.train() # Tryb treningowy
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad() # Zerowanie gradientów
            
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Obliczenie straty
            loss.backward() # Backward pass (obliczenie gradientów)
            optimizer.step() # Aktualizacja wag

            running_train_loss += loss.item() * inputs.size(0)
            
            # Obliczanie dokładności treningowej
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_train_loss / len(train_dataset)
        epoch_train_acc = correct_train / total_train

        # Walidacja
        model.eval() # Tryb ewaluacji (wyłącza np. Dropout, jeśli byłby w modelu)
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad(): # Wyłączamy śledzenie gradientów (oszczędność pamięci i czasu)
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = correct_val / total_val

        # Zapis historii
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # Wypisywanie progresu co 10 epok
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # D) Wizualizacja procesu uczenia
    plt.figure(figsize=(12, 5))

    # Wykres funkcji straty (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Trening Loss')
    plt.plot(history['val_loss'], label='Walidacja Loss')
    plt.title('Krzywa funkcji straty (Loss)')
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Wykres dokładności (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Trening Accuracy')
    plt.plot(history['val_acc'], label='Walidacja Accuracy')
    plt.title('Krzywa dokładności (Accuracy)')
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ==========================================
    # E) Końcowe statystyki i macierz błędu
    # ==========================================
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    final_accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*40)
    print("STATYSTYKI NA ZBIORZE WALIDACYJNYM:")
    print("="*40)
    print(f"Końcowe Accuracy: {final_accuracy * 100:.2f}%\n")
    print("Macierz błędu (Confusion Matrix):")
    print(conf_matrix)

    # Wizualizacja macierzy błędu
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Macierz błędu (Zbior Walidacyjny)')
    plt.show()

if __name__ == "__main__":
    main()