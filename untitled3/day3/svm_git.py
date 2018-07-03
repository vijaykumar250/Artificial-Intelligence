
# http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
# # -------------------------------------------------------------
# # regression with all the other types of regression models
# import
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import r2_score,mean_squared_error

classifiers1 = [
    linear_model.LinearRegression()]

classifiers2 = [
    svm.SVR(),
   linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor()
]

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# import dataset
data = pd.read_csv(r"https://raw.githubusercontent.com/DLMLPYRTRAINING/Day3/master/Datasets/Logistic_regression1.csv")

#Example: various way of fetching data from pandas
X = data.loc[0:len(data)-2, "Height"]
print(X)
X = data["Height"][:-1]
print(X)
X = data[:-1]["Height"]
print(X)

weight_val = [np.log(i/(1-i)) for i in data["Weight"].values]

# fetch the data
Train_F = data[:-1]["Height"].values.reshape(-1, 1)
print(Train_F)
plt.plot(Train_F, "b.")
Train_T = data[:-1]["Weight"].values.reshape(-1, 1)
print(Train_T)
plt.plot(Train_T, "bo")

Test_F = data[-1:]["Height"].values.reshape(-1, 1)
print(Test_F)
plt.plot(len(Train_F), Test_F, "g.")
Test_T = data[-1:]["Weight"].values.reshape(-1, 1)
print(Test_T)
plt.plot(len(Train_T), Test_T, "go")

for item in classifiers1:
    # build model
    model = item

    # train model
    model.fit(Train_F, Train_T)

    # score model
    score = model.score(Train_F, Train_T)

    # predict
    predict = model.predict(Test_F)

    #error%
    mse = mean_squared_error(Test_T,predict)
    r2 = r2_score(Test_T,predict)

    # print everything
    print(item)
    print("score\n", score)
    print("predict:\n", predict)
    print("actual:\n", Test_T)
    print("mean_squared_error:\n",mse)
    print("R2:\n",r2)
    time.sleep(5)

for item in classifiers2:
    # build model
    model = item

    # train model
    model.fit(Train_F,Train_T.ravel())
   #  print(model.support_vectors_)

    # score model
    score = model.score(Train_F, Train_T.ravel())

    # predict
    predict = model.predict(Test_F)

    #error%
    mse = mean_squared_error(Test_T.ravel(),predict)
    r2 = r2_score(Test_T.ravel(),predict)

    # print everything
    print(item)
    print("score\n", score)
    print("predict:\n", predict)
    print("actual:\n", Test_T)
    print("mean_squared_error:\n",mse)
    print("R2:\n",r2)


#show plot
plt.show()