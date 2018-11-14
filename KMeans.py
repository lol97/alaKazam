import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

data = pd.read_csv('data.csv')

f1 = data['sepallength'].values
f2 = data['sepalwidth'].values
f3 = data['petallength'].values
f4 = data['petalwidth'].values

X = np.array(list(zip(f1, f2, f3, f4)))

kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_

data1 = kmeans.predict(X)

for x in data1:
	print(x)

# data.head()