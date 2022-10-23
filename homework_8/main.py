import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]


def show_img(img, title = '', cmap = None):
    plt.title(title)
    plt.imshow(img, cmap = cmap)
    plt.show()


def show_two_imgs(img1, img2, title1 = '', title2 = '', cmap1 = None, cmap2 = None):
    plt.subplot(121), plt.title(title1), plt.imshow(img1, cmap = cmap1)
    plt.subplot(122), plt.title(title2), plt.imshow(img2, cmap = cmap2)
    plt.show()


def plot_histogram(h):
    plt.bar(h[1][0:-1], h[0])
    plt.xlabel('Colour'), plt.ylabel('Count')
    plt.grid(True)
    plt.show()


img = cv2.imread('resources/document.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_img(img_gray, 'Source image (gray)', 'gray')

# build histogram
h = np.histogram(img, bins = 256, range = (0, 255))
plot_histogram(h)

# Otsu Thresholding
# TODO finish
