import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris_big.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=298571)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

tree = DecisionTreeClassifier()
tree.fit(train_inputs, train_classes)
print(tree.score(test_inputs, test_classes))