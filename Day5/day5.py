import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn import datasets

def calculate_residuals(model, features, trueValue):
    predictions = model.predict(features)
    residuals = trueValue - predictions
    return residuals
    
boston = datasets.load_boston()

#Create a dataset that perfectly fits all assumptions
linear_X, linear_y = datasets.make_regression(n_samples=boston.data.shape[0],
                                              n_features=boston.data.shape[1],
                                              noise=75, random_state=46)

linear_feature_names = ['X'+str(feature+1) for feature in range(linear_X.shape[1])]

#Original boston dataset that doesn't fit assumptions
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['HousePrice'] = boston.target

print(df.head())

from sklearn.linear_model import LinearRegression

# Create models for both boston and generated linear dataset
boston_model = LinearRegression()
boston_model.fit(boston.data, boston.target)

linear_model = LinearRegression()
linear_model.fit(linear_X, linear_y)
 
#Linear Dependency Assumption
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
axes[0].scatter(linear_y , linear_model.predict(linear_X), color = 'red')
line_coords = np.arange(linear_y.min().min(), linear_y.max().max())
axes[0].plot(line_coords, line_coords, color='darkorange', linestyle='--')
axes[0].set_title('Linear generated')

axes[1].scatter(boston.target , boston_model.predict(boston.data), color = 'red')
line_coords = np.arange(boston.target.min().min(), boston.target.max().max())
axes[1].plot(line_coords, line_coords, color='darkorange', linestyle='--')
axes[1].set_title('Original Boston Dataset')

#Normality of residuals
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

residuals_linear = calculate_residuals(linear_model, linear_X, linear_y)
n, x,_ = axes[0].hist(residuals_linear, bins=15)
bin_centers = 0.5*(x[1:]+x[:-1])
axes[0].plot(bin_centers, n)

residuals_boston = calculate_residuals(boston_model, boston.data, boston.target)
n, x,_ = axes[1].hist(residuals_boston, bins=15)
bin_centers = 0.5*(x[1:]+x[:-1])
axes[1].plot(bin_centers, n)

from statsmodels.stats.diagnostic import normal_ad
print('Linear model: %s' % normal_ad(residuals_linear)[1])
print('Boston model: %s' % normal_ad(residuals_boston)[1])

#Multicolinearity
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

corr = pd.DataFrame(linear_X, columns=linear_feature_names).corr()

cax = axes.matshow(corr, interpolation='none')
axes.set_xticklabels(linear_feature_names)
axes.set_yticklabels(linear_feature_names)
axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
axes.yaxis.set_major_locator(ticker.MultipleLocator(1))
fig.colorbar(cax)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
corr2 = pd.DataFrame(boston.data, columns=boston.feature_names).corr()
cax = axes.matshow(corr2, interpolation='none')
axes.set_xticklabels(boston.feature_names)
axes.set_yticklabels(boston.feature_names)
axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
axes.yaxis.set_major_locator(ticker.MultipleLocator(1))
fig.colorbar(cax)

from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF_linear = [variance_inflation_factor(linear_X, i) for i in range(linear_X.shape[1])]
VIF_boston = [variance_inflation_factor(boston.data, i) for i in range(boston.data.shape[1])]
print('Linear VIF: %s' % VIF_linear)
print('Boston VIF: %s' % VIF_boston)

#Homoscedacity
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

axes[0].scatter(x= range(0, residuals_linear.size), y=residuals_linear, alpha=0.5)
axes[0].plot(np.repeat(0, residuals_linear.size), color='darkorange', linestyle='--')
    
axes[1].scatter(x= range(0, residuals_boston.size), y=residuals_boston, alpha=0.5)
axes[1].plot(np.repeat(0, residuals_boston.size), color='darkorange', linestyle='--')
plt.show()


    