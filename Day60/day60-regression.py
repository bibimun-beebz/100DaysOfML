import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X = np.sort(30 * np.random.rand(100, 1), axis=0)
print(X)
y = np.sin(X).ravel()

print(y)
y[::5] += 3 * (0.5 - np.random.rand(20))

y_true = np.sin(X).ravel()

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y, y_true))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=10)

layers = (100,) * 37
print(layers)

reg = MLPRegressor(hidden_layer_sizes=layers, max_iter=10000, alpha = 0.0002, learning_rate_init=0.0001, tol=1e-20, n_iter_no_change=50, verbose=True)

reg.fit(X_train, y_train)
x_plot = [np.arange(1,30, 0.1)]
plt.scatter(X_test , y_test, color = 'red')
plt.scatter(X_train , y_train, color = 'green')
plt.plot(X , y_true, color ='black',  linestyle='dashed')
plt.plot(np.transpose(x_plot) , reg.predict(np.transpose(x_plot)), color ='blue')

plt.show()

