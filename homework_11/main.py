import cv2
import numpy as np
import matplotlib
import pandas as pd
from numpy.random import seed
from tensorflow.python.framework.random_seed import set_seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Model, metrics
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Sequential
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]


def plot_history(history, coef):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122), plt.plot(epochs, np.array(h['mean_absolute_error']) * coef, '.-',
                               epochs, np.array(h['val_mean_absolute_error']) * coef, '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('MAE')
    plt.legend(['Train', 'Validation'])

    print('Train MAE     ', h['mean_absolute_error'][-1] * coef)
    print('Validation MAE', h['val_mean_absolute_error'][-1] * coef)

    plt.show()


# set the seeds
seed_value = 1234578790
seed(seed_value)
set_seed(seed_value)

# load dataset
dataset = pd.read_csv('resources/train.csv')

# Data Preparation

# get attributes as inputs
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
data = dataset[features]
# fill the missing values with the mean value of the dataset
data = data.fillna(data.mean())
# normalize data
inputs = data[features[1:]]
scale = StandardScaler()
inputs = scale.fit_transform(inputs)
outputs = data[features[0]].values
outputs = outputs / 100000
# split data into 75% for train and 25% for test
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size = 0.25)

# Building the Network
