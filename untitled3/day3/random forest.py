# classification - random forest
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel(r"C:\Users\TechPC\PycharmProjects\untitled3\day3\Classification_randomforest.xlsx")

#first we'll have to convert the strings "Single_Digit" and "Double_Digit" to numeric values
data.loc[data["Class"] == "Single_Digit", "Class"] = 0
data.loc[data["Class"] == "Double_Digit", "Class"] = 1
X = data["Number"][:-2].values.reshape(-1, 1)
Y = data["Class"][:-2].values.reshape(-1, 1)

x_test = data["Number"][-2:].values.reshape(-1, 1)
y_test = data["Class"][-2:].values.reshape(-1, 1)

# build model
model = RandomForestClassifier()

# train model
model.fit(X,Y.ravel())

# score model
score = model.score(X,Y.ravel())
print("Score:\n", score)

predict = model.predict(x_test)
print("Prediction Feature:\n", x_test)
print("Prediction Value:\n", predict)
print("Actual Value:\n",y_test)
