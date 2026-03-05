import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

pca_iris = PCA(n_components=4)
pca_iris.fit(X)

variance_ratios = pca_iris.explained_variance_ratio_

print("Wyjaśniona wariancja przez poszczególne kolumny (składowe):")
for i, ratio in enumerate(variance_ratios):
    print(f"Składowa {i}: {ratio:.4f} ({ratio*100:.2f}%)")

print("\nSkumulowana wariancja:")
print(np.cumsum(variance_ratios))

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA zbioru danych Iris (2 komponenty)')
plt.xlabel('Pierwsza składowa główna (PC1)')
plt.ylabel('Druga składowa główna (PC2)')

plt.savefig('pca_iris_2d.png')