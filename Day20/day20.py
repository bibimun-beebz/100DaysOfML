import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

def make_meshgrid(x, y, h=0.05):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def calcColour(label):
    if label == 0:
        return 'red'
    if label == 1:
        return 'blue'
    if label == 2:
        return 'green'
    return 'black'  
    
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn import decomposition 
pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)
X = pca.transform(iris.data)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, iris.target, test_size=0.3,random_state=109)
color= [calcColour(l) for l in iris.target]
xx, yy = make_meshgrid(X[:,0], X[:,1])
from sklearn.linear_model import Perceptron

model = Perceptron().fit(X_train, y_train)

print("Accuracy:",metrics.accuracy_score(y_test, model.predict(X_test)))
print("Coeff", model.coef_)

plot_contours(plt, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:,0], X[:,1], color=color)
plt.show()

