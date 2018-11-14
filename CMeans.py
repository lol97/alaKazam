import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

import skfuzzy as fuzz

data = pd.read_csv('data.csv')

f1 = data['sepallength'].values
f2 = data['sepalwidth'].values
f3 = data['petallength'].values
f4 = data['petalwidth'].values

X = np.array(list(zip(f1, f2, f3, f4))).T
# print(X.shape)
# kmeans = KMeans(n_clusters=3)
# kmeans = kmeans.fit(X)
# labels = kmeans.predict(X)
# centroids = kmeans.cluster_centers_

# data1 = kmeans.predict(X)

# print(data1)
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, 3, 2, error=0.005, 
	maxiter=1000, init=None)
# print(cntr.shape)

# data1, u1, d1, jm1, p1, fpc1 = fuzz.cluster.cmeans_predict(X, cntr, 3, 
# 	error=0.005, maxiter=1000, init=None)

cluster_membership = np.argmax(u, axis=0)
# print(len(cluster_membership))
# print(cluster_membership)
for x in cluster_membership:
	print(x)




# print(data1)
# print(u1)
# print(d1)
# print(jm1)
# print(p1)
# print(fpc1)
# print(cntr.shape)
# print("ini yang 0: ", cntr[0])
# print("ini yang 1: ", cntr[1])
# print("ini yang 2: ", cntr[2])

# data.head()