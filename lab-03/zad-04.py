import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv('diagnosis.csv')
X = df.iloc[:, :-1]  # Wszystkie wiersze, wszystkie kolumny oprócz ostatniej
y = df.iloc[:, -1]   # Wszystkie wiersze, tylko ostatnia kolumna

# TRÓJWYMIAROWY WYKRES PUNKTOWY
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Rozdzielenie na chorych i zdrowych do legendy
zdrowi = df[df.iloc[:, -1] == 0]
chorzy = df[df.iloc[:, -1] == 1]

# Rysowanie punktów
ax.scatter(zdrowi.iloc[:, 0], zdrowi.iloc[:, 1], zdrowi.iloc[:, 2], c='blue', label='Zdrowi (0)', alpha=0.6)
ax.scatter(chorzy.iloc[:, 0], chorzy.iloc[:, 1], chorzy.iloc[:, 2], c='red', label='Chorzy (1)', alpha=0.6)

ax.set_xlabel('Parametr 1')
ax.set_ylabel('Parametr 2')
ax.set_zlabel('Parametr 3')
ax.set_title('Wykres 3D: Parametry medyczne a diagnoza')
ax.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

klasyfikatory = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "3-NN": KNeighborsClassifier(n_neighbors=3),
    "5-NN": KNeighborsClassifier(n_neighbors=5),
    "11-NN": KNeighborsClassifier(n_neighbors=11),
    "Naive Bayes": GaussianNB(),
    "MLP (Sieć Neuronowa)": MLPClassifier(max_iter=2000, random_state=42)
}

# Ustawiamy siatkę wykresów dla macierzy błędów (2 rzędy, 3 kolumny)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (nazwa, clf) in enumerate(klasyfikatory.items()):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Obliczenie metryk
    # zero_division=0 zapobiega błędom, gdy model nie przewidzi żadnego przypadku danej klasy
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"--- {nazwa} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}\n")
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Przew. Zdrowy', 'Przew. Chory'],
                yticklabels=['Fakt. Zdrowy', 'Fakt. Chory'])
    axes[idx].set_title(f"{nazwa}\nAcc: {acc:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f}")

plt.tight_layout()
plt.show()

# Czy Accuracy jest bezpieczną miarą przy niezbalansowanym zbiorze?
# Absolutnie NIE. Zjawisko to nazywa się paradoksem dokładności (accuracy paradox).
# Wyobraź sobie, że w bazie masz 99 osób zdrowych i 1 osobę chorą. Jeśli stworzysz beznadziejny, "leniwy" model, który zawsze i każdemu mówi "jesteś zdrowy" (nie analizując
# nawet danych), to jego Accuracy wyniesie 99% (zgadł 99 zdrowych, pomylił się tylko na jednym chorym). 99% brzmi super, ale model w rzeczywistości jest całkowicie bezużyteczny
# , bo przeoczył jedynego chorego (jego Recall wynosi 0%).