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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=10)

reg = MLPRegressor(hidden_layer_sizes=(100,100, 100, 100, 100, 100, 100,100,100,100, 100,100,100,100,100,100,100,100,100,100,100), max_iter=3000, learning_rate_init=0.0001, tol=1e-9, verbose=True)

reg.fit(X_train, y_train)
x_plot = [np.arange(1,30, 0.1)]
plt.scatter(X , y, color = 'red')
plt.plot(np.transpose(x_plot) , reg.predict(np.transpose(x_plot)), color ='blue')
plt.show()