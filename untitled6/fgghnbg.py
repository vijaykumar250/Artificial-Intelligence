# import models
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt

# Prepare your data
# data_file_path = "E:\ArtificialIntelligence&MachineLearning\MachineLearning\Day3\linearxl.xlsx"
data = pd.read_excel('E:/ArtificialIntelligence&MachineLearning/MachineLearning/Day3/LINEAR3.xlsx')

# incase of missing data impute data
# data['x] = data['x'].fillna(np.nanmean(data['x']))


# # split the data into training and test
feature_train = data['EngineFuelCapacity'][:-1].values
feature_train = np.reshape(feature_train,(-1,1))
print(feature_train)
plt.plot(feature_train,'b.')

feature_test = data['Horsepower'][-1:].values
feature_test = np.reshape(feature_test,(-1,1))
print(feature_test)
plt.plot(len(feature_train),feature_test,'bo')

# --

target_train = data['EngineFuelCapacity'][:-1].values
target_train = np.reshape(target_train,(-1,1))
print(target_train)
plt.plot(target_train,'g.')

target_test = data['Horsepower'][-1:].values
target_test = np.reshape(target_test,(-1,1))
print(target_test)
plt.plot(len(target_train),target_test,'go')

# build model
model = LinearRegression()

# train model with your data
model.fit(feature_train,target_train)

# Score your model
score = model.score(feature_train,target_train)
print("Score:\n",score)

# predict data using your model
target_prediction = model.predict(feature_test)
print("Prediction:\n",target_prediction)
plt.plot(len(feature_train),target_prediction,'rx')

# calculate the amount of error in your prediction
MSE = mean_squared_error(target_test,target_prediction)
R2 = r2_score(target_test,target_prediction)
print("MSE:\n",MSE)
print("R2:\n",R2)

# print statistics for your model
intercept = model.intercept_
coeff = model.coef_
print("Intercept:\n",intercept)
print("Coeff:\n",coeff)

# plot the graph based on the intercept and coefficient
points = [intercept+coeff[0]*eachitem[0] for eachitem in feature_train]
print(points)
plt.plot(points,'r--')

#legendd






# plot model

plt.show()

