# iris_cluster

import numpy as n
import pandas as p
import matplotlib.pyplot as plt
import sklearn.metrics as so
from sklearn.cluster import KMeans
from sklearn import datasets as d


iris=d.load_iris()
print(iris.data)
print(iris.target)
print(type(iris.target))#dependent variable
x=p.DataFrame(iris.data)
##print(type(x))
x.columns=["sepallength",'sepalwidth','petallength','petalwidth']
colormap=n.array(['red','lime','black'])
y=p.DataFrame(iris.target)
##print(y)
y.columns=['Target']
plt.subplot(1,2,1)
plt.figure(figsize=(14,7))
plt.scatter(x.sepallength,x.sepalwidth,c=colormap[y.Target],s=40)
plt.title("Sepal Data")
plt.show()
plt.figure(figsize=(14,7))
plt.scatter(x.petallength,x.petalwidth,c=colormap[y.Target])
plt.title("petal data")
plt.show()
model = KMeans(n_clusters=3)
model.fit(x)
centroids = model.cluster_centers_
print("centroids",centroids)
labels=model.labels_
print(labels)
plt.figure(figsize=(14,7))
plt.scatter(x.petallength,x.petalwidth,c=colormap[y.Target])
plt.title("petal data after model")
plt.show()
pred_y=n.choose(labels,[1,0,2])
print(labels)
print(pred_y)
print(so.accuracy_score(y,pred_y))
print(so.confusion_matrix(y,pred_y))


