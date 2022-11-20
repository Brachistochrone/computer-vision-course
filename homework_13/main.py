import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy.random import seed
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]


def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-',
                               epochs, h['val_accuracy'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    print('Train Acc     ', h['accuracy'][-1])
    print('Validation Acc', h['val_accuracy'][-1])

    plt.show()


# set the seeds
seed_value = 1234578790
seed(seed_value)
set_seed(seed_value)

# Dataset Loading

# load CIFAR-10 dataset
# CIFAR-10 consists of 60000 32x32 colour images in 10 classes with about 6000 images per class
(in_train, out_train), (in_test, out_test) = tf.keras.datasets.cifar10.load_data()
classes = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
num_classes = len(classes)
size = in_train.shape[1]
# visualize random samples
for ii in range(18):
    idx = np.random.randint(0, len(in_train))
    plt.subplot(3, 6, ii + 1), plt.imshow(in_train[idx, ...])
    plt.title(classes[int(out_train[idx])])
plt.show()
# compute the class histogram
centers = np.arange(0, num_classes + 1)
counts, bounds = np.histogram(out_train.flatten(), bins = centers - 0.5)
plt.bar(centers[:-1], counts), plt.grid(True)
plt.xlabel('Class ID'), plt.ylabel('counts')
plt.show()

# Data Preparation

# normalization
in_train = in_train / 255
in_test = in_test / 255
# One-hot encoding
out_train = tf.keras.utils.to_categorical(out_train, num_classes)
out_test = tf.keras.utils.to_categorical(out_test, num_classes)
print('Train set:   ', len(out_train), 'samples')
print('Test set:    ', len(out_test), 'samples')
print('Sample dims: ', in_train.shape)

# Building Classifier

# 32x32 RGB
inputs = Input(shape = (size, size, 3))
# 16 3x3 kernels
net = Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = 'same')(inputs)
net = MaxPooling2D(pool_size = (2, 2))(net)
# 32 3x3 kernels
net = Conv2D(32, kernel_size = (3, 3), activation = "relu", padding = 'same')(net)
net = MaxPooling2D(pool_size = (2, 2))(net)
net = Flatten()(net)
outputs = Dense(num_classes, activation = "softmax")(net)
model = Model(inputs, outputs)
print(model.summary())

# Training

epochs = 25
batch_size = 128
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
history = model.fit(in_train, out_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
plot_history(history)

# Evaluation

out_true = np.argmax(out_test, axis = 1)
# test model
out_pred = model.predict(in_test)
out_pred = np.argmax(out_pred, axis = 1)
# calculate accuracy of prediction for each digit
for class_id, class_name in classes.items():
    mask = out_true == class_id
    tp = np.sum(out_pred[mask] == class_id)
    total = np.sum(mask)
    print('Class name:', class_name, ' acc:', tp / total)
# print the overall stats
ev = model.evaluate(in_test, out_test)
print('Test loss  ', ev[0])
print('Test metric', ev[1])
# show random outcome samples
for ii in range(15):
    idx = np.random.randint(0, len(in_test))
    plt.subplot(3, 5, ii + 1), plt.imshow(in_test[idx, ...])
    plt.title('True: ' + str(classes[out_true[idx]]) + ' | Pred: ' + str(classes[out_pred[idx]]))
plt.show()

# Answers:
# - the overall accuracy of the classifier is about 0.68
# - in order to improve prediction accuracy, I would increase epoch number,
#   since the history graphics show that the learning curves have not reached flat regions yet
#   also, I guess using AlexNet architecture for this classifier would be beneficial too
#   adding more layers to the current network does not help much
# - increasing the number of epochs to 30 slightly improves the prediction accuracy,
#   but the problem of over-fitting becomes more obvious
