from sklearn import datasets, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

digits = datasets.load_digits()
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
    
plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

print(digits.images[0])

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.6, shuffle=True, random_state=1)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic',
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1, verbose=True)

mlp.fit(X_train,y_train)

predicted = mlp.predict(X_test)

print(predicted)
print(y_test)
wrong_ones = []
for index, p in enumerate(predicted):
    if(y_test[index] != p):
        wrong_ones.append(index)
        
print(wrong_ones)

#hidden_2 = np.transpose(mlp.coefs_[0])[32]  # Pull weightings on inputs to the 2nd neuron in the first hidden layer
hidden_2 = np.transpose(mlp.coefs_[0])[5]  # Pull weightings on inputs to the 2nd neuron in the first hidden layer

print(hidden_2)
fig, ax = plt.subplots(1, figsize=(5,5))
ax.imshow(np.reshape(hidden_2, (8,8)), cmap=plt.get_cmap("gray"), aspect="auto")

_, axes = plt.subplots(nrows=1, ncols=len(wrong_ones), figsize=(10, 3))
for ax, image, prediction, label in zip(axes, X_test[wrong_ones], predicted[wrong_ones], y_test[wrong_ones]):
    ax.set_axis_off()
    ax.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'{prediction}/{label}')
    
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
print(predicted[wrong_ones[30]])
plt.bar(np.arange(10), predicted[wrong_ones[30]])
plt.xticks(np.arange(10))
plt.show()