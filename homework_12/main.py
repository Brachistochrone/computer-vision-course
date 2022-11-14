import cv2
import numpy as np
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]

root = '/data/janko/dataset/GTSRB'
data = pd.read_csv(os.path.join(root, 'Train.csv'))

num_samples =

# show random samples
for ii in range(15):
    # get random index
    idx = np.random.randint(0, num_samples)
    # load image
    img = cv2.imread(os.path.join(root, data.iloc[idx]['Path']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # show image
    plt.subplot(3, 5, ii + 1), plt.imshow(img), plt.title(data.iloc[idx]['ClassId'])
