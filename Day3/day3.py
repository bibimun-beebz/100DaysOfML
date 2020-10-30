import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("One Hot Encode", OneHotEncoder(),[3])], remainder="passthrough")
X = ct.fit_transform(X)

X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print(Y_pred)
print(Y_test)
