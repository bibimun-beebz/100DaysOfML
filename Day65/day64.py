from sklearn import datasets, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math
import pandas as pd


digits = datasets.load_digits()
#_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
#for ax, image, label in zip(axes, digits.images, digits.target):
#    ax.set_axis_off()
#    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#    ax.set_title('Training: %i' % label)
#    
#plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

print(digits.images[0])

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.6, shuffle=True, random_state=1)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(60,), activation='logistic',
                   tol=1e-4, random_state=1, alpha = 0.2,
                    learning_rate_init=.01, verbose=True)

mlp.fit(X_train,y_train)

predicted = mlp.predict(X_test)

print(predicted)
print(y_test)
wrong_ones = []
for index, p in enumerate(predicted):
    if(y_test[index] != p):
        wrong_ones.append(index)
        
print(wrong_ones)
rows = len(wrong_ones)//10
cols = math.ceil(len(wrong_ones)/rows)

hidden_2 = np.transpose(mlp.coefs_[0])[32]  # Pull weightings on inputs to the 2nd neuron in the first hidden layer
hidden_2 = np.transpose(mlp.coefs_[0])[5]  # Pull weightings on inputs to the 2nd neuron in the first hidden layer

print(hidden_2)
fig, ax = plt.subplots(nrows=6, ncols=10, figsize=(5,5))
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for index, node in enumerate(np.transpose(mlp.coefs_[0])):
    ax[index % 6, index//6].imshow(np.reshape(node, (8,8)), cmap=plt.get_cmap("gray"), aspect="auto", vmin=.5 * vmin, vmax=.5 * vmax)

_, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 3))
for index, (image, prediction, label) in enumerate(zip(X_test[wrong_ones], predicted[wrong_ones], y_test[wrong_ones])):
    axes[index % rows, index // rows].imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray_r, interpolation='nearest')
    axes[index % rows, index // rows].set_title(f'{prediction}/{label}')

fig, ax = plt.subplots(1, 1, figsize=(15,6))
ax.imshow(np.transpose(mlp.coefs_[0]), cmap=plt.get_cmap("gray"), aspect="auto")
    
    
print(f"Classification report for classifier {mlp}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

disp = metrics.plot_confusion_matrix(mlp, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
plt.plot(mlp.loss_curve_)
plt.show()


predicted = mlp.predict_proba(X_test)
print( len(wrong_ones))

fig, axs = plt.subplots(rows, cols)

for index, wrong_index in enumerate(wrong_ones):
    axs[index % rows, index // rows].bar(np.arange(10), predicted[wrong_index])
    axs[index % rows, index // rows].set_xticks(np.arange(10))
    axs[index % rows, index // rows].set_yticks(np.arange(0,1,0.1))
    axs[index % rows, index // rows].set_title(f'{y_test[wrong_index]}')
plt.show()