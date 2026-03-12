import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris_big.csv")

#podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = [298571]
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=298571)

def classify_iris(sl, sw, pl, pw):
 if sl > 4.4 and sl < 5.5:
  return("setosa")
 elif pl > 4.9 and pl < 6.5:
  return("virginica")
 else:
   return("versicolor")
 
good_predictions = 0
len = test_set.shape[0]
for i in range(len):
 if classify_iris(test_set[i][0], test_set[i][1], test_set[i][2], test_set[i][3]) == test_set[i][4]:
  good_predictions = good_predictions + 1
print(good_predictions)
print(good_predictions/len*100, "%")

# PIERWSZA PRÓBA: 356
# 79.11111111111111 %

# train_set_sorted = train_set.sort_values(by='target_name')
# print(train_set_sorted)
# print(train_set)
# print(test_set)

# train_inputs = train_set[:, 0:4]
# train_classes = train_set[:, 4]
# test_inputs = test_set[:, 0:4]
# test_classes = test_set[:, 4]

# print(test_set)
# print(test_set.shape[0])

# print(train_inputs)
# print(train_classes)
# print(test_inputs)
# print(test_classes)