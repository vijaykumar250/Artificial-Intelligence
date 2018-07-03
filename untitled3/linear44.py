import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

#import datasets
df = pd.read_csv("https://raw.githubusercontent.com/DLMLPYRTRAINING/Day3/master/Datasets/Linear4.csv")

#impute the data
df.fillna(0,inplace=True)
#print(df)
#
# #training X and Y
mean_x = np.mean(df["number_of_seeds"].values)
print(mean_x)
mean_y = np.mean(df["number_of_fruit"].values)
print(mean_y)
df['number_of_seeds'] = df['number_of_seeds'].replace(0, mean_x)
df['number_of_fruit'] = df['number_of_fruit'].replace(0, mean_y)
#df.fillna(df.mean()['number_of_seeds':'number_of_fruit'])
print(df)

train_X = df["number_of_seeds"][:-1].values
train_Y = df["number_of_fruit"][:-1].values

train_X =np.reshape(train_X,(-1,1))
train_Y = np.reshape(train_Y,(-1,1))

#testing X and Y
test_X = df["number_of_seeds"][-1:].values
test_Y = df["number_of_fruit"][-1:].values
test_X =np.reshape(test_X,(-1,1))
test_Y = np.reshape(test_Y,(-1,1))


#build the model
model = LinearRegression()

#fit the model
model.fit(train_X,train_Y)
coef = model.coef_
intercept = model.intercept_
print("coefficient:",coef)
print("intercept:",intercept)

point = [intercept + coef[0]*i[0] for i in train_X]
plt.plot(point,'r--')

#score the model
score = model.score(train_X,train_Y)
print("Score:",score)

#predict the model
predict = model.predict(test_X)
print("prediction:",predict)

print("RMSE:",mean_squared_error(test_Y,predict))
print("R2:",r2_score(test_Y,predict))
#plotting
plt.plot(train_X,train_Y,)
plt.plot(len(df),test_Y,"go")
plt.plot(len(df),predict,"rx")
plt.xlabel('no. of seeds')
plt.ylabel('no. of fruits')
plt.show()