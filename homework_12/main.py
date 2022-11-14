import cv2
import numpy as np
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]

root = 'D:/GTSRB'
data = pd.read_csv(os.path.join(root, 'Train.csv'))

ids = data.get('ClassId')
num_samples = len(ids)
classes = set(ids)
num_classes = len(classes)

# show random samples
for ii in range(15):
    # get random index
    idx = np.random.randint(0, num_samples)
    # load image
    img = cv2.imread(os.path.join(root, data.iloc[idx]['Path']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # show image
    plt.subplot(3, 5, ii + 1), plt.imshow(img), plt.title(data.iloc[idx]['ClassId'])
plt.show()

# build histogram of class distribution in dataset
centers = np.arange(0, num_classes + 1)
counts, bounds = np.histogram(ids, bins = centers - 0.5)
# absolute histogram
plt.bar(centers[:-1], counts), plt.grid(True)
plt.xlabel('Class ID'), plt.ylabel('counts')
plt.show()
# percentage histogram
plt.bar(centers[:-1], counts / np.sum(counts) * 100), plt.grid(True)
plt.xlabel('Class ID'), plt.ylabel('%')
plt.show()

# Answers:
# - the dataset seems to be unbalanced because some classes have lots of samples and others have just few of them
#   as a result, the histogram of training samples does not look smooth
#   (however, the reason might be that I managed to unpack only half of the archive)
# the most overrepresented classes are: 1, 2, 4, 10, 12, 13, and 38 that have over 35% of all samples
# the most underrepresented classes are: 0, 19, 24, 27, 29, 32, 37, 41, and 42 that have around 0.5% of all samples each
