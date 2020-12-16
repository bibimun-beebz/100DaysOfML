import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X = np.sort(30 * np.random.rand(100, 1), axis=0)
print(X)
y = np.sin(X).ravel()

print(y)
y[::5] += 3 * (0.5 - np.random.rand(20))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#scores_list = []
#i_range = range(1,10000,100)
#for i in i_range:
#    print(i)
#    reg = GradientBoostingRegressor(n_estimators=i, learning_rate=1, max_depth=1, random_state=0)
#    reg.fit(X_train, y_train)
#    mse = mean_squared_error(y_test, reg.predict(X_test))
#    scores_list.append(mse)
#
#
#plt.plot(i_range, scores_list)
#plt.show()


reg = GradientBoostingRegressor(n_estimators=50, learning_rate=1, max_depth=1, random_state=0)
reg.fit(X_train, y_train)
x_plot = [np.arange(1,30, 0.1)]
print(np.shape(np.transpose(x_plot)))
print(np.shape(X))
plt.scatter(X , y, color = 'red')
plt.plot(np.transpose(x_plot) , reg.predict(np.transpose(x_plot)), color ='blue')
plt.show()