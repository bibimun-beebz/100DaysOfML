import pandas as pd
import numpy as np

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values

print(dataset)
print(X)
print(Y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "mean")
print()
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
print('After label')
print(X)

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("One Hot Encode", OneHotEncoder(),[0])], remainder="passthrough")
X = ct.fit_transform(X)

labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

print('One hot')
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_train)
print(X_test)
