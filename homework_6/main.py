import cv2
import math
import matplotlib
import numpy as np
from sklearn.cluster import KMeans
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


def get_descriptor(target_quadrant, q2, q3, q4):
    return np.mean(target_quadrant) - np.mean(q2) - np.mean(q3) - np.mean(q4)


def highlight_corner(image, corner_coordinates):
    cv2.circle(image, corner_coordinates, 5, (255, 0, 0), -1)


img = cv2.imread('resources/document.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = img_gray.astype(np.float32) / 255
rows, columns = img_gray.shape
show_two_imgs(img, img_gray, 'Original', 'Gray scale', cmap2 = 'gray')

# find corners using Harris Corner Detector
corners = cv2.cornerHarris(img_gray, blockSize = 2, ksize = 3, k = 0.04)
# eliminate edges (negative values)
corners[corners < 0] = 0
# reduce dynamic range by taking logarithm for better visualization and manipulation
corners = np.log(corners + 1e-6)

show_two_imgs(img, corners, 'Original', 'Cornerness')

# detection thresholds
threshold_top_left, threshold_top_right, threshold_bottom_left, threshold_bottom_right = -1e6, -1e6, -1e6, -1e6
# detected corner coordinates
top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner = None, None, None, None
# size of each quadrant
quad_size = 7
# scan the Harris detection results
for row in range(quad_size, rows - quad_size):
    for column in range(quad_size, columns - quad_size):
        # skip pixels with too small cornerness (-7 threshold)
        if corners[row, column] < -7:
            continue
        # pick block of four quadrants around a pixel (14x14)
        block = img_gray[row - quad_size: row + quad_size + 1, column - quad_size:column + quad_size + 1]
        # extract quadrants
        quad_top_left = block[0:quad_size, 0:quad_size]
        quad_top_right = block[0:quad_size, quad_size + 1:quad_size * 2 + 1]
        quad_bottom_left = block[quad_size + 1:quad_size * 2 + 1, 0:quad_size]
        quad_bottom_right = block[quad_size + 1:quad_size * 2 + 1, quad_size + 1:quad_size * 2 + 1]
        # calculate descriptors fro corners
        # in the end there will be corner coordinates with max descriptors
        # Top-left corner -> bottom right quadrant has max value -> biggest descriptor
        descriptor = get_descriptor(quad_bottom_right, quad_top_left, quad_top_right, quad_bottom_left)
        if descriptor > threshold_top_left:
            threshold_top_left = descriptor
            top_left_corner = (column, row)
        # Top-right corner -> bottom left quadrant has max value -> biggest descriptor
        descriptor = get_descriptor(quad_bottom_left, quad_top_left, quad_top_right, quad_bottom_right)
        if descriptor > threshold_top_right:
            threshold_top_right = descriptor
            top_right_corner = (column, row)
        # Bottom-left corner -> top right quadrant has max value -> biggest descriptor
        descriptor = get_descriptor(quad_top_right, quad_top_left, quad_bottom_right, quad_bottom_left)
        if descriptor > threshold_bottom_left:
            threshold_bottom_left = descriptor
            bottom_left_corner = (column, row)
        # Bottom-right corner -> top left quadrant has max value -> biggest descriptor
        descriptor = get_descriptor(quad_top_left, quad_top_right, quad_bottom_right, quad_bottom_left)
        if descriptor > threshold_bottom_right:
            threshold_bottom_right = descriptor
            bottom_right_corner = (column, row)

out = np.copy(img)
highlight_corner(out, top_left_corner)
highlight_corner(out, top_right_corner)
highlight_corner(out, bottom_left_corner)
highlight_corner(out, bottom_right_corner)
show_two_imgs(img, out, 'Original', 'Highlighted corners')

# Answers:
# -
