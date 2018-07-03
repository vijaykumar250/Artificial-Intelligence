from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Prepare your data
data_file_path = 'E:\ArtificialIntelligence&MachineLearning\MachineLearning\Day3\sampledata.xlsx'
data = pd.read_excel(data_file_path)

# incase of missing data
data['Weight'] = data['Weight'].fillna(np.nanmean(data['Weight']))

# # split the data into training and test
feature_train = data['Height'][:-1].values
feature_train = np.reshape(feature_train,(-1,1))
print(feature_train)
plt.plot(feature_train,'b.')

feature_test = data['Height'][-1:].values
feature_test = np.reshape(feature_test,(-1,1))
print(feature_test)
plt.plot(len(feature_train),feature_test,'bo')

# --------------------------------------------------

target_train = data['Weight'][:-1].values
target_train = np.reshape(target_train,(-1,1))
print(target_train)
plt.plot(target_train,'g.')

target_test = data['Weight'][-1:].values
target_test = np.reshape(target_test,(-1,1))
print(target_test)
plt.plot(len(target_train),target_test,'go')

# build model
model = LinearRegression()

# train model with your data
#model.fit(feature_train,target_train)

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


# plot model

plt.show()
