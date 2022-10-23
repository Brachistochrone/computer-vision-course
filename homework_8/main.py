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
img_gray = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
show_img(img_gray, 'Source image (gray)', 'gray')

# build histogram
h = np.histogram(img_gray, bins = 256, range = (0, 255))
plot_histogram(h)

# Otsu Thresholding
rows, columns = img_gray.shape
num_pixels = rows * columns
# Best within-class variance (wcv)
best_wcv = 1e6
# Threshold corresponding to the best wcv
opt_th = None
# Brute force search using all possible thresholds
for th in range(0, 256):
    # get foreground pixels
    foreground = img_gray[img_gray > th]
    # get background pixels
    background = img_gray[img_gray <= th]
    if len(foreground) == 0 or len(background) == 0:
        continue
    # compute class-weights (omega parameters) for foreground and background
    omega_f = len(foreground) / num_pixels
    omega_b = 1 - omega_f
    # compute pixel variance for foreground and background (result ^2 already)
    sigma_f = np.var(foreground)
    sigma_b = np.var(background)
    # compute within-class variance
    wcv = np.sqrt(omega_f * sigma_f + omega_b * sigma_b)
    if wcv < best_wcv:
        best_wcv = wcv
        opt_th = th

print('Optimal threshold', opt_th)
show_two_imgs(img_gray, img_gray > opt_th, 'Original', 'Optimal threshold', 'gray', 'gray')

# Answers:
# - based on the histogram, I wouldn't say that the picture has bimodal distribution because
#   even though there is a distinct cluster of bright pixels, but there is no distinct cluster of dark pixels
#   and dark pixels are more or less evenly distributed
# - I would pick 175..177 as an optimal threshold because it's where the slope of bright pixel cluster begins to rise
# - resulting text binarization is not good enough since white letters are barely seen
