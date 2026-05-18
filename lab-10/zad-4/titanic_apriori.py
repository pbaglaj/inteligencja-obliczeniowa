import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

print("--- ANALIZA REGUŁ ASOCJACYJNYCH: TITANIC ---")

# ==========================================
# 1. Wczytanie i przygotowanie danych
# ==========================================
# index_col=0 pomija pierwszą kolumnę (numerację wierszy)
try:
    df = pd.read_csv('titanic.csv', index_col=0)
except FileNotFoundError:
    print("Błąd: Nie znaleziono pliku titanic.csv.")
    exit()

# Ograniczamy do 4 głównych kolumn (na wypadek gdyby w pliku było coś więcej)
df = df[['Class', 'Sex', 'Age', 'Survived']]

# TŁUMACZENIE NA JĘZYK POLSKI (dla idealnej czytelności na wykresie i w konsoli)
df['Class'] = df['Class'].map({'1st': '1. Klasa', '2nd': '2. Klasa', '3rd': '3. Klasa', 'Crew': 'Załoga'})
df['Sex'] = df['Sex'].map({'Male': 'Mężczyzna', 'Female': 'Kobieta'})
df['Age'] = df['Age'].map({'Adult': 'Dorosły', 'Child': 'Dziecko'})
df['Survived'] = df['Survived'].map({'No': 'Zginął', 'Yes': 'Przeżył'})

# One-Hot Encoding
# prefix='' i prefix_sep='' usuwają przedrostki kolumn. 
# Zamiast "Sex_Mężczyzna" otrzymamy po prostu cechę "Mężczyzna".
df_encoded = pd.get_dummies(df, prefix='', prefix_sep='').astype(bool)

print(f"Rozmiar danych po przygotowaniu: {df_encoded.shape}\n")

# ==========================================
# 2. Algorytm Apriori i wyszukiwanie reguł
# ==========================================
# Generowanie zbiorów częstych (min_support = 0.005)
frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)

# Generowanie reguł asocjacyjnych (min_confidence = 0.8)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# Sortowanie malejąco według ufności (confidence)
rules = rules.sort_values(by='confidence', ascending=False)
print(f"Znaleziono ogółem {len(rules)} reguł spełniających kryteria.\n")

# ==========================================
# 3. Wyszukiwanie "najciekawszych" reguł
# ==========================================
# Filtrujemy tylko te reguły, w których skutkiem jest "Przeżył" lub "Zginął"
def filters_survival(consequents):
    return any(item in ['Przeżył', 'Zginął'] for item in consequents)

survival_rules = rules[rules['consequents'].apply(filters_survival)]

print("--- NAJCIEKAWSZE REGUŁY DOTYCZĄCE PRZEŻYCIA (Top 10) ---")
for index, row in survival_rules.head(10).iterrows():
    # Czyste formatowanie z przecinkami
    antecedents = ", ".join(list(row['antecedents']))
    consequents = ", ".join(list(row['consequents']))
    print(f"JEŚLI [{antecedents}] TO [{consequents}] | "
          f"Ufność: {row['confidence']:.2f}, Wsparcie: {row['support']:.3f}, Lift: {row['lift']:.2f}")

# ==========================================
# 4. Zobrazowanie reguł na wykresie
# ==========================================
plt.figure(figsize=(11, 7))
scatter = plt.scatter(
    survival_rules['support'], 
    survival_rules['confidence'], 
    c=survival_rules['lift'], 
    cmap='viridis', 
    alpha=0.8,
    s=120, 
    edgecolors='w'
)

plt.colorbar(scatter, label='Przyrost (Lift) - siła reguły')
plt.xlabel('Wsparcie (Support) - jak dużej części pasażerów to dotyczy')
plt.ylabel('Ufność (Confidence) - prawdopodobieństwo reguły')
plt.title('Reguły asocjacyjne na Titanicu (kto przeżył, a kto zginął)')
plt.grid(True, linestyle='--', alpha=0.5)

# Podpisujemy 5 najważniejszych punktów na wykresie czytelnym tekstem
top_rules = survival_rules.head(5)
for i, row in top_rules.iterrows():
    ant_text = "\n".join(list(row['antecedents'])) # Wypisuje cechy pod sobą
    cons_text = list(row['consequents'])[0]
    
    plt.annotate(
        f"{ant_text} \n➔ {cons_text}",
        (row['support'], row['confidence']),
        xytext=(8, -10), textcoords='offset points',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8) # Estetyczne ramki pod tekstem
    )

plt.tight_layout()
plt.show()