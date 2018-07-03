# classification with logistic regression
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("https://raw.githubusercontent.com/DLMLPYRTRAINING/Day3/master/Datasets/Logistic_classification1.csv")

#first we'll have to convert the strings "No" and "Yes" to numeric values
data.loc[data["default"] == "No", "default"] = 0
data.loc[data["default"] == "Yes", "default"] = 1
X = data["balance"][:-1].values.reshape(-1, 1)
Y = data["default"][:-1].values.reshape(-1, 1)

x_test = data["balance"][-1:].values.reshape(-1, 1)
y_test = data["default"][-1:].values.reshape(-1, 1)

LogR = LogisticRegression()
LogR.fit(X, np.ravel(Y.astype(int)))
score = LogR.score(X, np.ravel(Y.astype(int)))
print("Score:\n", score)

coeff = LogR.coef_
intercept = LogR.intercept_
print("Coeff\n", coeff)
print("Intercept\n", intercept)

predict = LogR.predict(x_test)
predict_proba_class = LogR.predict_proba(x_test)
print("Prediction Feature:\n", x_test)
print("Prediction Value:\n", predict)
print("Actual Value:\n",y_test)
print("Prediction Class:\n", predict_proba_class)

def model_plot(x):
    return 1/(1+np.exp(-x))

points = [intercept+coeff*i for i in X.ravel()]
points = np.ravel([model_plot(i) for i in points])
plt.plot(points,'g')

#matplotlib scatter funcion w/ logistic regression
plt.plot(X,'rx')
plt.plot(Y,'bo')
plt.xlabel("Credit Balance")
plt.ylabel("Probability of Default")
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
plt.legend(["Logistic Regression Model","X","Y"],
           loc="lower right", fontsize='small')
plt.show()
