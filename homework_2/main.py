# Colour Balancing
import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]


def show_two_imgs(img1, img2):
    plt.subplot(121), plt.imshow(img1)
    plt.subplot(122), plt.imshow(img2)
    plt.show()


def apply(src_img, rgb_coeffs):
    new_img = np.zeros_like(src_img, dtype = np.float32)
    chs = 3
    # apply balancing
    for ch in range(chs):
        new_img[..., ch] = src_img[..., ch] * rgb_coeffs[ch]
    # balancing does not guarantee that the dynamic range is preserved, images must be clipped
    new_img = new_img / 255
    new_img[new_img > 1] = 1
    return new_img


img = cv2.imread('resources/dark_sea.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# White patch
rows, columns, channels = img.shape
# find the most white pixel - 'white patch'
rgb_sum, w_row, w_column = 0, 0, 0
for row in range(rows):
    for column in range(columns):
        if np.sum(img[row, column]) > rgb_sum:
            rgb_sum = np.sum(img[row, column])
            w_row = row
            w_column = column
# define coefficients needed to make 'white patch' actually white
white_patch = img[w_row, w_column, :]
coeffs = 255.0 / white_patch
# apply white balancing
balanced = apply(img, coeffs)
show_two_imgs(img, balanced)

# Gray world
# assumes that in a normal well color balanced photo, the average of all the colors is a neutral gray
# compute mean values for all three colour channels
mean_rgb = np.mean(img, axis = (0, 1))
# calculate coefficients for every channel
kr = np.max(mean_rgb) / mean_rgb[0]
kg = np.max(mean_rgb) / mean_rgb[1]
kb = np.max(mean_rgb) / mean_rgb[2]
coeffs = (kr, kg, kb)
# apply coefficients to every channel
balanced = apply(img, coeffs)
show_two_imgs(img, balanced)

# Scale-by-max
# increases levels of colours so that the brightest value of each channel becomes 255
# compute maximum values for all three colour channels
max_rgb = np.max(img, axis = (0, 1))
coef_rgb = 255 / max_rgb
balanced = apply(img, coef_rgb)
show_two_imgs(img, balanced)
