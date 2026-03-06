import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

iris_df = pd.read_csv("iris_big.csv")
target_names = iris_df['target_name'].unique()

X_original = iris_df[['sepal length (cm)', 'sepal width (cm)']]
y = iris_df['target_name']

# Z-Score (StandardScaler)
z_scaler = StandardScaler()
X_zscore = z_scaler.fit_transform(X_original)

# Min-Max (MinMaxScaler)
mm_scaler = MinMaxScaler()
X_minmax = mm_scaler.fit_transform(X_original)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ['Original Dataset', 'Z-Score Scaled Dataset', 'Min-Max Normalised Dataset']
datasets = [X_original.values, X_zscore, X_minmax]
y_labels = ['Sepal Width (cm)', 'Sepal Width (cm)', 'Sepal Width (cm)']
x_labels = ['Sepal Length (cm)', 'Sepal Length (cm)', 'Sepal Length (cm)']

colors = ['tab:blue', 'tab:orange', 'tab:green']

for i, ax in enumerate(axes):
    for label, color in zip(target_names, colors):
        mask = y == label  # porównujemy tekst z tekstem (np. 'setosa' == 'setosa')
        ax.scatter(datasets[i][mask, 0], datasets[i][mask, 1], 
                   c=color, label=label, edgecolors='white', alpha=0.8)
    
    ax.set_title(titles[i])
    ax.set_xlabel(x_labels[i])
    ax.set_ylabel(y_labels[i])
    ax.legend()

plt.tight_layout()
plt.show()

print(X_original.describe().loc[['min', 'max', 'mean', 'std']])