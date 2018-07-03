# classification - decision tree

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
print(cross_val_score(clf, iris.data, iris.target, cv=10))
# -----------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

data = pd.read_excel(r"C:\Users\TechPC\PycharmProjects\untitled3\day3\Classification_randomforest.xlsx")
#first we'll have to convert the strings "Single_Digit" and "Double_Digit" to numeric values
data.loc[data["Class"] == "Single_Digit", "Class"] = 0
data.loc[data["Class"] == "Double_Digit", "Class"] = 1
X = data["Number"][:-2].values.reshape(-1, 1)
Y = data["Class"][:-2].values.reshape(-1, 1)

x_test = data["Number"][-2:].values.reshape(-1, 1)
y_test = data["Class"][-2:].values.reshape(-1, 1)

# build model
model = DecisionTreeClassifier()
# train model
model.fit(X,Y)

# score model
score = model.score(X,Y)
print("Score:\n", score)

predict = model.predict(x_test)
print("Prediction Feature:\n", x_test)
print("Prediction Value:\n", predict)
print("Actual Value:\n",y_test)

