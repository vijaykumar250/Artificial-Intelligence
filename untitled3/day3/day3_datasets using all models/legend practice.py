import numpy as np
import pandas as pd
import pandas
import matplotlib.pyplot as plt
data = pandas.read_excel('sampledata.xlsx')
print(data.shape)
data.head()

# Collecting X and Y
X = data['lift1_True'].values
Y = data['lift2_Predict'].values
 
# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values
m = len(X)

# Using the formula to calculate b1 and b2
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

# Print coefficients
print(b1, b0)
# Plotting Values and Regression Line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('list1')
plt.ylabel('list2')
plt.legend()
plt.show()

# Calculating Root Mean Squares Error
rmse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/m)
print(rmse)
# Calculating  Mean Squares Error MSE
mse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    mse += (Y[i] - y_pred) ** 2
print(mse)
# calculating r2
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)

# The coefficients
print('Coefficients: \n', b1,b0)
# The root  mean squared error
print('ROOT Mean squared error: \n',rmse)
# The root  mean squared error
print('Mean squared error: \n',mse)      
# Explained variance score: 1 is perfect prediction
print('Variance score: ',r2)

# Plot outputs
plt.scatter(X, Y,  color='black')
plt.plot(X, Y, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cannot use Rank 1 matrix in scikit learn
X = X.reshape((m, 1))
# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# Calculating RMSE and R2 Score
mse = mean_squared_error(Y, Y_pred)
print(mse)
rmse = np.sqrt(mse)
print(rmse)
r2_score = reg.score(X, Y)


print(r2_score)




