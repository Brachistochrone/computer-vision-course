import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras import Model, metrics
from tensorflow.python.keras.layers import Dense, Input

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]


def prepare_and_normalize(dataset, keys):
    # get attributes as inputs
    data = dataset[keys]
    # fill the missing values with the mean value of the dataset
    data = data.fillna(data.mean())
    # normalize data
    x = data[keys[1:]]
    scale = StandardScaler()
    x = scale.fit_transform(x)
    y = data[keys[0]].values
    y = y / 100000
    # split data into 75% for train and 25% for test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
    return x_train, x_test, y_train, y_test


def build_network(input_num, layer_num = 1, neuron_num = 1, title = ''):
    # configure input layer
    ins = Input(shape = input_num)
    # add layers
    layers = None
    for ln in range(layer_num):
        if ln == 0:
            layers = Dense(neuron_num, activation = 'relu')(ins)
        else:
            layers = Dense(neuron_num, activation = 'relu')(layers)
    # configure output layer
    layers = Dense(1, activation = 'linear')(layers)
    # create network
    network = Model(ins, layers)
    network._name = title
    # compile network
    network.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = [metrics.mae])
    return network


def plot_history(history, coef, name = ''):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation']), plt.title(name + ' MSE')
    plt.subplot(122), plt.plot(epochs, np.array(h['mean_absolute_error']) * coef, '.-',
                               epochs, np.array(h['val_mean_absolute_error']) * coef, '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('MAE')
    plt.legend(['Train', 'Validation']), plt.title(name + ' MAE')

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
inputs_train, inputs_test, outputs_train, outputs_test = prepare_and_normalize(dataset, features)

# Building the Network

# create network (one layers 5 neurons)
network = build_network(len(features) - 1, 1, 5, 'Initial_network')
network.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = [metrics.mae])
print(network.summary())

# Training the Network

# train network
history = network.fit(inputs_train, outputs_train, validation_data = (inputs_test, outputs_test), epochs = 150,
                      batch_size = 32, verbose = 0)
plot_history(history, 1e5, 'Initial network')

# Improving the Network

# add more input parameters (+'LotArea')
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'LotArea']
inputs_train, inputs_test, outputs_train, outputs_test = prepare_and_normalize(dataset, features)
network = build_network(len(features) - 1, 1, 5, 'Network_with_more_inputs')
print(network.summary())
history = network.fit(inputs_train, outputs_train, validation_data = (inputs_test, outputs_test), epochs = 150,
                      batch_size = 32, verbose = 0)
plot_history(history, 1e5, 'Network with more inputs')

# increase the number of neurons in the layer (1x10)
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
inputs_train, inputs_test, outputs_train, outputs_test = prepare_and_normalize(dataset, features)
network = build_network(len(features) - 1, 1, 10, 'Network_with_more_neurons')
print(network.summary())
history = network.fit(inputs_train, outputs_train, validation_data = (inputs_test, outputs_test), epochs = 150,
                      batch_size = 32, verbose = 0)
plot_history(history, 1e5, 'Network with more neurons')

# increase the number of layers (2x5)
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
inputs_train, inputs_test, outputs_train, outputs_test = prepare_and_normalize(dataset, features)
network = build_network(len(features) - 1, 2, 5, 'Network_with_more_layers')
print(network.summary())
history = network.fit(inputs_train, outputs_train, validation_data = (inputs_test, outputs_test), epochs = 150,
                      batch_size = 32, verbose = 0)
plot_history(history, 1e5, 'Network with more layers')

# Answers:
# - improvements are not equally helpful
# - adding one more input parameter doesn't make the network any better (actually the outcome becomes even worse)
# - doubling the number of neurons in the intermediate layer shows the best result
# - adding one more layer also slightly improves the network's outcome
# - however, neither modification alone shows a drastic improvement in the network's performance;
#   perhaps, to achieve the best results, all of them should be combined
