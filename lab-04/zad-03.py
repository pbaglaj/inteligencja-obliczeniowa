import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    ConfusionMatrixDisplay
)

# 1. DATA LOADING & PREPROCESSING

class DiagnosisDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. NEURAL NETWORK ARCHITECTURE

class DiagnosisNet(nn.Module):
    def __init__(self, input_features, num_classes):
        super(DiagnosisNet, self).__init__()
        """
        Dynamic MLP Architecture:
        - Input layer matches the number of features in diagnosis.csv
        - Hidden layers use ReLU activation
        - Output layer matches the number of unique diagnosis classes
        """
        self.fc1 = nn.Linear(input_features, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 3. MAIN TRAINING & EVALUATION LOOP

def main():
    try:
        df = pd.read_csv('../diagnosis.csv', header=0)
    except FileNotFoundError:
        print("Error: 'diagnosis.csv' not found")
        return

    X_raw = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    # Encode textual labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    num_features = X_raw.shape[1]
    num_classes = len(np.unique(y_encoded))

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    except ValueError:
        print("Warning: Stratification failed (likely due to rare classes). Splitting randomly.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_raw, y_encoded, test_size=0.2, random_state=42
        )

    # Feature Scaling (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create DataLoaders
    batch_size = 16
    train_loader = DataLoader(DiagnosisDataset(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DiagnosisDataset(X_val_scaled, y_val), batch_size=batch_size, shuffle=False)

    # Initialize Model, Loss function, and Optimizer
    model = DiagnosisNet(input_features=num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 100
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val

        # Save metrics for plotting
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # 4. EVALUATION & VISUALIZATION
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- Final Metrics (Accuracy, Precision, Recall, Confusion Matrix) ---
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics (using 'weighted' to handle potential class imbalance)
    final_acc = accuracy_score(all_labels, all_preds)
    final_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    final_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*50)
    print("FINAL VALIDATION METRICS:")
    print("="*50)
    print(f"Accuracy:  {final_acc * 100:.2f}%")
    print(f"Precision: {final_precision * 100:.2f}%")
    print(f"Recall:    {final_recall * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Oranges)
    plt.title('Confusion Matrix (Validation Set)')
    plt.show()

if __name__ == "__main__":
    main()