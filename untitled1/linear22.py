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
