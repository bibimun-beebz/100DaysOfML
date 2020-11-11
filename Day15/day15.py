from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics

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
print("Features: ", iris.feature_names)
print("Labels: ", iris.target_names)
print(iris.data[0:5])

from sklearn import decomposition 
pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)
X = pca.transform(iris.data)
print(pca.explained_variance_ratio_)
#plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_, color='green')
#plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, iris.target, test_size=0.3,random_state=109)
color= [calcColour(l) for l in iris.target]
xx, yy = make_meshgrid(X[:,0], X[:,1])
from sklearn.ensemble import BaggingClassifier

clf = BaggingClassifier(base_estimator=svm.SVC(kernel='poly', C=1, degree=10), n_estimators=30, random_state=0, max_samples=0.6).fit(X_train, y_train)
print('score')
print(metrics.accuracy_score(y_test,  clf.predict(X_test)))

from sklearn import metrics
p_range = range(1,20)
scores = {}
scores_list = []
for p in p_range:
    clf = svm.SVC(kernel='poly', degree=p) 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores[p] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(p_range, scores_list)
plt.xlabel('Value of degree for SVM poly')
plt.ylabel('Testing Accuracy')
plt.show()

p_range =[1, 10, 100, 1000]
scores = {}
scores_list = []
for p in p_range:
    clf = svm.SVC(kernel='rbf', C=p) 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores[p] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(p_range, scores_list)
plt.xlabel('Value of C for SVM RBF')
plt.ylabel('Testing Accuracy')
plt.show()

from sklearn.model_selection import GridSearchCV
params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
          'degree': range(1,10),
          'kernel':['poly'] }

grid_clf = GridSearchCV(svm.SVC(), params_grid, cv=10)
grid_clf.fit(X_train, y_train)
print(grid_clf.best_params_)

clf = svm.SVC(kernel='poly', degree=grid_clf.best_params_['degree']) 
clf.fit(X_train, y_train)
print('score')
print(clf.score(X_test, y_test))
    
plot_contours(plt, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:,0], X[:,1], color=color)
plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], color='black')
plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, clf.predict(X_test)))
