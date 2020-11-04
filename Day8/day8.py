import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Standardising helps to reduce the impact of multicolinearity
from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='none', max_iter=10000, verbose=1, tol=1)
classifier.fit(X_train, y_train)
print(classifier.coef_)

from sklearn import metrics

#metrics.plot_roc_curve(classifier, X_test, y_test)

#print(metrics.r2_score(y_test, classifier.predict(X_test)))


#print( X_test[:,0])
#probabilities= []
#for i in X_test[:,0]:
#    p_loss, p_win = classifier.predict_proba([[i]])[0]
#    probabilities.append(p_win)
#    
#print(probabilities)

#plt.scatter(X_test[:,0], probabilities)
#plt.scatter(X_test[:,0], y_test)
#plt.show()

#from sklearn.metrics import plot_confusion_matrix
#plot_confusion_matrix(classifier, X_test, y_test)
#plt.show()

