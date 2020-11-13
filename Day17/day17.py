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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, iris.target, test_size=0.3,random_state=109)
color= [calcColour(l) for l in iris.target]
xx, yy = make_meshgrid(X[:,0], X[:,1])
model = GaussianNB()
model.fit(X_train, y_train);

plot_contours(plt, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:,0], X[:,1], color=color)
plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, model.predict(X_test)))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

print(model.theta_)
print(model.sigma_)
print(model.classes_)
print(model.theta_[1,0])
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-10, 10, 0.001)
# Mean = 0, SD = 2.
plt.plot(x_axis, norm.pdf(x_axis,model.theta_[0,0],math.sqrt(model.sigma_[0,0])), color=calcColour(0))
plt.plot(x_axis, norm.pdf(x_axis,model.theta_[1,0],math.sqrt(model.sigma_[1,0])), color=calcColour(1))
plt.plot(x_axis, norm.pdf(x_axis,model.theta_[2,0],math.sqrt(model.sigma_[2,0])), color=calcColour(2))
plt.show()

plt.plot(x_axis, norm.pdf(x_axis,model.theta_[0,1],math.sqrt(model.sigma_[0,1])), color=calcColour(0))
plt.plot(x_axis, norm.pdf(x_axis,model.theta_[1,1],math.sqrt(model.sigma_[1,1])), color=calcColour(1))
plt.plot(x_axis, norm.pdf(x_axis,model.theta_[2,1],math.sqrt(model.sigma_[2,1])), color=calcColour(2))
plt.show()
#MultinomialNB
from sklearn.datasets import fetch_20newsgroups

print('fetching data...')
categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print(train.data[5])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]
    
print(predict_category('sending a payload to the ISS'))

