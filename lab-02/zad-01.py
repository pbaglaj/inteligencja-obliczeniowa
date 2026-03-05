import pandas as pd
from pathlib import Path

#
# TRZEBA DOROBIC KOREKCJE SLOW!!!
# I TO Sprawdź czy wszystkie dane numeryczne są z zakresu (0; 15). Dane spoza zakresu muszą być poprawione. Możesz
# tutaj użyć metody: za błędne dane podstaw średnią (lub medianę) z danej kolumny.
#

base_path = Path(__file__).parent
file_path = base_path / "iris_big_with_errors.csv"

df = pd.read_csv(file_path, names=range(15), dtype=str, on_bad_lines="warn")

total_rows = len(df)

cond_5_cols = df[4].notna() & df[5].isna()

regex_pattern = r'^\d+\.\d{2}$'

cond_c0 = df[0].str.match(regex_pattern, na=False)
cond_c1 = df[1].str.match(regex_pattern, na=False)
cond_c2 = df[2].str.match(regex_pattern, na=False)
cond_c3 = df[3].str.match(regex_pattern, na=False)

valid_species = ['versicolor', 'setosa', 'virginica']
cond_c4 = df[4].isin(valid_species)

valid_mask = cond_5_cols & cond_c0 & cond_c1 & cond_c2 & cond_c3 & cond_c4

df_clean = df[valid_mask].copy()

df_clean = df_clean.iloc[:, :5]

df_clean.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

for col in df_clean.columns[:4]:
    df_clean[col] = df_clean[col].astype(float)

good_rows_count = len(df_clean)
bad_rows_count = total_rows - good_rows_count

print(f"Łączna liczba wierszy: {total_rows}")
print(f"Liczba usuniętych (błędnych) wierszy: {bad_rows_count}")
print(f"Liczba pozostawionych (poprawnych) wierszy: {good_rows_count}")
