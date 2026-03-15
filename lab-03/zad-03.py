import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Wczytanie danych
df = pd.read_csv('iris_big.csv')
X = df.drop('target_name', axis=1)
y = df['target_name']

STATE = 298571

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=STATE)

# Definiujemy wszystkie klasyfikatory w jednym słowniku
# Ustawiamy random_state dla powtarzalności. Dla MLP zwiększamy max_iter, aby sieć zdążyła się nauczyć.
klasyfikatory = {
    "Drzewo decyzyjne (Zad 2)": DecisionTreeClassifier(random_state=STATE),
    "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "k-NN (k=11)": KNeighborsClassifier(n_neighbors=11),
    "Naive Bayes": GaussianNB(),
    "Sieć Neuronowa (MLP)": MLPClassifier(max_iter=1000, random_state=STATE)
}

wyniki_dokladnosci = {}

for nazwa, clf in klasyfikatory.items():
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    dokladnosc = accuracy_score(y_test, y_pred) * 100
    wyniki_dokladnosci[nazwa] = dokladnosc
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"--- {nazwa} ---")
    print(f"Dokładność: {dokladnosc:.2f}%")
    print("Macierz błędów:")
    print(cm)
    print("-" * 30 + "\n")

# Sortujemy wyniki malejąco po dokładności
posortowane_wyniki = sorted(wyniki_dokladnosci.items(), key=lambda item: item[1], reverse=True)

for miejsce, (nazwa, dokladnosc) in enumerate(posortowane_wyniki, start=1):
    print(f"{miejsce}. {nazwa}: {dokladnosc:.2f}%")