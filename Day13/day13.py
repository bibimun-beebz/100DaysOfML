from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def make_meshgrid(x, y, h=1):
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
    

cancer = datasets.load_breast_cancer()
print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)
print(cancer.data[0:5])

from sklearn import decomposition 
pca = decomposition.PCA(n_components=2)
pca.fit(cancer.data)
X = pca.transform(cancer.data)
print(pca.explained_variance_ratio_)
plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_, color='green')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, cancer.target, test_size=0.3,random_state=109)
from sklearn import svm

clf = svm.SVC(kernel='linear') 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

color= [calcColour(l) for l in cancer.target]
xx, yy = make_meshgrid(X[:,0], X[:,1])
plot_contours(plt, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:,0], X[:,1], color=color)
plt.show()

clf = svm.SVC(kernel='rbf') 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_contours(plt, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:,0], X[:,1], color=color)
plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


clf = svm.SVC(kernel='poly') 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plot_contours(plt, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:,0], X[:,1], color=color)
plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
