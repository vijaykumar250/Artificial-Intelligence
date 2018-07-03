# Clustering - Kmeans
# import
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# X = np.array([1,2,3,4,5,6,20,21,22,23,24,50,51,52,53,54,55])
# X = X.reshape(-1,1)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print(kmeans.labels_)

print(kmeans.predict([[0, 0], [4, 4]]))
# print(kmeans.predict([[9], [30], [60]]))

print(kmeans.cluster_centers_)

plt.plot(X[:,0],X[:,1],'bx')
plt.plot(0,0,'rx')
plt.plot(4,4,'rx')
plt.plot(1,2,'gx')
plt.plot(4,2,'gx')
plt.show()
