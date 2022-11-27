from random import randint

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy.random import seed as np_seed
from sklearn.utils import shuffle
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D

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


def rotate_image(image, rot_mat):
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags = cv2.INTER_LINEAR)
    return result


def get_batch_augmented(x, y, batch_size):
    num_samples = len(y)
    while True:
        for idx in range(0, num_samples, batch_size):
            # get batch
            x_ = x[idx:idx + batch_size, ...]
            y_ = y[idx:idx + batch_size]
            # if it's the end of data set -> shuffle it and start again
            if len(y_) < batch_size:
                x, y = shuffle(x, y)
                break
            # random mirroring
            for idx_aug in range(batch_size):
                if randint(0, 1) == 1:
                    x_[idx_aug, ...] = np.fliplr(x_[idx_aug, ...])
            yield x_, y_


# set the seeds
seed_value = 1234578790
np_seed(seed_value)
set_seed(seed_value)

# Dataset Loading

# load MNIST fashion dataset - monochrome images of different types of clothing
(in_train, out_train), (in_test, out_test) = tf.keras.datasets.fashion_mnist.load_data()

num_classes = 10
size = in_train.shape[1]

print('Train set:   ', len(out_train), 'samples')
print('Test set:    ', len(out_test), 'samples')
print('Sample dims: ', in_train.shape)

# show random samples
cnt = 1
for r in range(3):
    for c in range(6):
        idx = randint(0, len(in_train))
        plt.subplot(3, 6, cnt)
        plt.imshow(in_train[idx, ...], cmap = 'gray')
        plt.title(out_train[idx])
        cnt = cnt + 1
plt.show()

# Building the Classifier

# normalize data
in_train = in_train / 255
in_test = in_test / 255
# build the network
# 28x28 monochrome
inputs = Input(shape = (28, 28, 1))
net = Conv2D(32, kernel_size = (3, 3), activation = "relu", padding = 'same')(inputs)
net = Flatten()(net)
net = Dense(128)(net)
outputs = Dense(10, activation = "softmax")(net)
model = Model(inputs, outputs)
print(model.summary())
# train the network
epochs = 50
batch_size = 64
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
history = model.fit(in_train, out_train, batch_size = batch_size, epochs = epochs,
                    validation_data = (in_test, out_test))
# classifier suffers from massive over-fitting
plot_history(history)

# Tackling over-fitting

# first - simplifying the network + adding dropouts
inputs = Input(shape = (28, 28, 1))
# reduce number of kernels (32 -> 28)
net = Conv2D(28, kernel_size = (3, 3), activation = "relu", padding = 'same')(inputs)
# add max pooling 2x2
net = MaxPooling2D(pool_size = (2, 2))(net)
# add 20% dropout
net = Dropout(0.2)(net)
net = Flatten()(net)
# reduce number of neurons (128 -> 64)
net = Dense(64)(net)
# add aggressive 50% dropout
net = Dropout(0.5)(net)
outputs = Dense(10, activation = "softmax")(net)
model = Model(inputs, outputs)
print(model.summary())
# train the network
epochs = 50
batch_size = 64
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
history = model.fit(in_train, out_train, batch_size = batch_size, epochs = epochs,
                    validation_data = (in_test, out_test))
plot_history(history)

# second - data augmentation + adding dropouts
inputs = Input(shape = (28, 28, 1))
net = Conv2D(32, kernel_size = (3, 3), activation = "relu", padding = 'same')(inputs)
net = MaxPooling2D(pool_size = (2, 2))(net)
net = Dropout(0.2)(net)
net = Flatten()(net)
net = Dense(128)(net)
net = Dropout(0.5)(net)
outputs = Dense(10, activation = "softmax")(net)
model = Model(inputs, outputs)
print(model.summary())
epochs = 50
batch_size = 64
steps_per_epoch = len(out_train) // batch_size
batch = get_batch_augmented(in_train, out_train, batch_size)
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
history = model.fit(batch, steps_per_epoch = steps_per_epoch, epochs = epochs,
                    validation_data = (in_test, out_test))
plot_history(history)

# Answers:
# - Firstly, I tried simplifying the network as well as adding dropouts and max pooling.
#   It did help to reduce over-fitting. Then, I returned the size of the network and used
#   randomly augmented data for training (horizontal mirroring) which showed the best results.
# - Also, I tried image rotation for 10...15 degrees as data augmentation, which turned out to be a disaster
#   because for some reason the network refused to learn at all. The training accuracy was gradually decreasing
#   whereas the validation accuracy fell dramatically
