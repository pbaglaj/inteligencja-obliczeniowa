import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

iris_df = pd.read_csv("iris_big.csv")

X = iris_df.iloc[:, 0:4]
y_labels = iris_df.iloc[:, 4]

# Konwertujemy gatunki tekstowe na numery (0, 1, 2) dla wykresu
y = pd.factorize(y_labels)[0]
target_names = np.unique(y_labels)

pca_full = PCA(n_components=4)
pca_full.fit(X)

print("Wyjaśniona wariancja przez poszczególne kolumny:")
for i, ratio in enumerate(pca_full.explained_variance_ratio_):
    print(f"Składowa {i}: {ratio:.4f} ({ratio*100:.2f}%)")

print("\nSkumulowana wariancja:")
print(np.cumsum(pca_full.explained_variance_ratio_))

pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X)

plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)

plt.legend(loc='best', shadow=False)
plt.title('PCA zbioru danych Iris (2 komponenty)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.savefig('pca_iris_2d.png')
print("\nWykres został zapisany jako 'pca_iris_2d.png'")