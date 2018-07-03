# import sklearn.linear_model
# import numpy as np
# from matplotlib import pyplot as plt
# import pandas as pd
#
# # file path
# data_file = "Data/data.csv"
#
# # pull the data from the file
# data = pd.read_csv(data_file, sep=',')
# train_features = data['X'][:-1].values
# train_features = np.reshape(train_features,(-1,1))
# # plot the features
# plt.plot(train_features, 'g.')
#
# train_target = data['Expected_output'][:-1].values
# train_target = np.reshape(train_target,(-1,1))
# # plot the targets
# plt.plot(train_target, 'go')
#
# test_features = data['X'][-1:].values
# test_features = np.reshape(test_features,(-1,1))
# # plot the targets
# plt.plot(len(train_features),test_features,'b.')
#
# test_target = data['Expected_output'][-1:].values
# test_target = np.reshape(test_target,(-1,1))
# # plot the targets
# plt.plot(len(train_target),test_target,'bo')
#
# # build the model
# model = sklearn.linear_model.LinearRegression()
#
# # fit data to model
# model.fit(train_features, train_target)
#
# # first level test the model with the data it was trained with
# # just to check if it's failing anywhere to predict anything through scoring
# score = model.score(train_features, train_target)
#
# # get some statistics out of the model
# coefficient = model.coef_
# intercept = model.intercept_
#
# print("score:\n", score)
# print("coeff:\n", coefficient)
# print("intercept:\n", intercept)
#
# # plot the linear regression line
# plt.plot([intercept + coefficient[0]*i[0] for i in train_features], 'r--')
#
# # lets do actual prediction
# prediction = model.predict(test_features)
# print("prediction:\n",prediction)
# # plot the prediction on top of the expected outcome
# plt.plot(len(train_target),prediction,'rx')
#
# # show the final plot
# plt.show()

# # -------------------------------------------------------------------------------------
# # 2 feature scenario
# import sklearn.linear_model
# import numpy as np
# from matplotlib import pyplot as plt
# import pandas as pd
#
# # file path
# data_file = "Data/data1.csv"
#
# # pull the data from the file
# data = pd.read_csv(data_file, sep=',')
# train_features= data[['X','Y']][:-1].values
# # train_features = np.reshape(train_features_before_reshaping,(-1,2))
# # plot the features
# plt.plot(train_features, 'g.')
#
# train_target = data['Expected_output'][:-1].values
# # train_target = np.reshape(train_target,(-1,2))
# # plot the targets
# plt.plot(train_target, 'go')
#
# test_features = data[['X','Y']][-1:].values
# # test_features = np.reshape(test_features,(-1,2))
# # plot the targets
# plt.plot(len(train_features),test_features,'b.')
#
# test_target = data['Expected_output'][-1:].values
# # test_target = np.reshape(test_target,(-1,2))
# # plot the targets
# plt.plot(len(train_target),test_target,'bo')
#
# # build the model
# model = sklearn.linear_model.LinearRegression()
#
#
# # fit data to model
# model.fit(train_features, train_target)
#
# # first level test the model with the data it was trained with
# # just to check if it's failing anywhere to predict anything through scoring
# score = model.score(train_features, train_target)
#
# # get some statistics out of the model
# coefficient = model.coef_
# intercept = model.intercept_
#
# print("score:\n", score)
# print("coeff:\n", coefficient)
# print("intercept:\n", intercept)
#
#
# # plot the linear regression line
# plt.plot([intercept + coefficient[0]*i[0] + coefficient[1]*i[1] for i in train_features], 'r--')
#
# # lets do actual prediction
# prediction = model.predict(test_features)
# print("prediction:\n",prediction)
# # plot the prediction on top of the expected outcome
# plt.plot(len(train_target),prediction,'rx')
#
#
# # finally show the plot
# plt.show()

























# ===================================================
# Scenario: 1 Feature1
# import
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Prepare your data
data_file_path = 'E:\ArtificialIntelligence&MachineLearning\MachineLearning\Day3\sampledata.xlsx'
data = pd.read_excel(data_file_path)
# print(data)
# print(data['Height'].values)

# incase of missing data
#data['Weight'] = data['Weight'].fillna(np.nanmean(data['Weight']))

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

# Save model
#joblib.dump(model,"E:\ArtificialIntelligence&MachineLearning\MachineLearning\Day3\neww")


# plot model

plt.show()
# ===================================================
# Scenario: Multi Feature
# ===================================================
# Scenario: imputing missing data
