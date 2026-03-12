import pandas as pd
from pathlib import Path

base_path = Path(__file__).parent
file_path = base_path / "iris_big_with_errors.csv"

df = pd.read_csv(file_path, names=range(15), dtype=str, on_bad_lines="warn")
total_rows = len(df)

df = df.iloc[:, :5]
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target_name']


# print(f"Łączna liczba wierszy wczytanych z pliku: {total_rows}")
# print("\nLiczba całkowicie pustych miejsc:")
# print(df.isna().sum())

for col in df.columns[:4]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


for col in df.columns[:4]:
    error_mask = (df[col] <= 0) | (df[col] >= 15) | df[col].isna()
    
    median_val = df.loc[~error_mask, col].median()
    
    df.loc[error_mask, col] = median_val

# print("Lista unikalnych wpisów gatunków przed korektą:")
# print(df['target_name'].unique())

def fix_species(name):
    name = str(name).lower().strip() 
    
    if 'set' in name:
        return 'setosa'
    elif 'vers' in name or 'ver' in name:
        return 'versicolor'
    elif 'virg' in name:
        return 'virginica'
    else:
        return 'setosa'

df['target_name'] = df['target_name'].apply(fix_species)

# print("\nLista unikalnych gatunków PO korekcie:")
# print(df['target_name'].unique())

# print("Brakujące dane (powinno być 0):")
# print(df.isna().sum())
# print("\nStatystyki opisowe wyczyszczonej bazy:")
# print(df.describe())