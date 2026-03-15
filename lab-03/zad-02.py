import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

df = pd.read_csv("iris_big.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=298571)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

tree = DecisionTreeClassifier()
tree.fit(train_inputs, train_classes)
# print(tree.score(test_inputs, test_classes))

# drzewo w formie tekstowej
tekst_drzewa = export_text(tree, feature_names=list(df.columns[:-1]))
# print(tekst_drzewa)
  
# drzewo w formie graficznej
plt.figure(figsize=(15, 10))
plot_tree(tree, filled=True, feature_names=list(df.columns[:-1]), class_names=list(tree.classes_))
# plt.show()

# Ewaluacja klasyfikatora
dokladnosc = tree.score(test_inputs, test_classes)
print(f"\nDokładność klasyfikatora na zbiorze testowym: {dokladnosc * 100:.2f}%")

# Macierz błędów (Confusion Matrix)
y_pred = tree.predict(test_inputs)
cm = confusion_matrix(test_classes, y_pred)
# print(cm)

# macierz błędów w formie graficznej
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()