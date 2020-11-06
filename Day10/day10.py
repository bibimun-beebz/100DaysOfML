from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)

print(iris.data.shape)
print(iris.target)
print(X_train)
print(y_train)

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
k_range = range(1,80)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
 
import matplotlib.pyplot as plt
plt.plot(k_range, scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

from sklearn import metrics
p_range = range(1,10)
scores = {}
scores_list = []
for p in p_range:
    knn = KNeighborsClassifier(p=p)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    scores[p] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
 
plt.plot(p_range, scores_list)
plt.xlabel('Value of P for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(iris.data)
X = pca.transform(iris.data)
labl = iris.target
plt.bar(['PC1', 'PC2', 'PC3', 'PC4'], pca.explained_variance_ratio_, color='green')
plt.show()

def calcColour(label):
    if label == 0:
        return 'red'
    if label == 1:
        return 'blue'
    if label == 2:
        return 'green'
    return 'black'    
    
color= [calcColour(l) for l in labl]
plt.scatter(X[:,0], X[:,1], color=color)
plt.show()
