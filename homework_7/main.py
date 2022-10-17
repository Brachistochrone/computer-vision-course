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


def get_descriptor(target_quadrant, q2, q3, q4):
    return np.mean(target_quadrant) - np.mean(q2) - np.mean(q3) - np.mean(q4)


def highlight_corners(image, corner_coordinates, radius):
    for corner in corner_coordinates:
        cv2.circle(image, corner, radius, (255, 0, 0), -1)


# load image
img = cv2.imread('resources/document.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = img_gray.astype(np.float32) / 255
rows, columns = img_gray.shape
show_two_imgs(img, img_gray, 'Original', 'Gray scale', cmap2 = 'gray')

# find coordinates of corners of the document
cornerness = cv2.cornerHarris(img_gray, blockSize = 2, ksize = 3, k = 0.04)
cornerness[cornerness < 0] = 0
cornerness = np.log(cornerness + 1e-6)
threshold_top_left, threshold_top_right, threshold_bottom_left, threshold_bottom_right = -1e6, -1e6, -1e6, -1e6
top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner = None, None, None, None
quad_size = 7
for row in range(quad_size, rows - quad_size):
    for column in range(quad_size, columns - quad_size):
        if cornerness[row, column] < -7:
            continue
        block = img_gray[row - quad_size: row + quad_size + 1, column - quad_size:column + quad_size + 1]
        quad_top_left = block[0:quad_size, 0:quad_size]
        quad_top_right = block[0:quad_size, quad_size + 1:quad_size * 2 + 1]
        quad_bottom_left = block[quad_size + 1:quad_size * 2 + 1, 0:quad_size]
        quad_bottom_right = block[quad_size + 1:quad_size * 2 + 1, quad_size + 1:quad_size * 2 + 1]
        descriptor = get_descriptor(quad_bottom_right, quad_top_left, quad_top_right, quad_bottom_left)
        if descriptor > threshold_top_left:
            threshold_top_left = descriptor
            top_left_corner = (column, row)
        descriptor = get_descriptor(quad_bottom_left, quad_top_left, quad_top_right, quad_bottom_right)
        if descriptor > threshold_top_right:
            threshold_top_right = descriptor
            top_right_corner = (column, row)
        descriptor = get_descriptor(quad_top_right, quad_top_left, quad_bottom_right, quad_bottom_left)
        if descriptor > threshold_bottom_left:
            threshold_bottom_left = descriptor
            bottom_left_corner = (column, row)
        descriptor = get_descriptor(quad_top_left, quad_top_right, quad_bottom_right, quad_bottom_left)
        if descriptor > threshold_bottom_right:
            threshold_bottom_right = descriptor
            bottom_right_corner = (column, row)

# print found corners
out = np.copy(img)
highlight_corners(out, (top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner), 5)
show_two_imgs(img, out, 'Original', 'Highlighted corners')

# define the matrix of source points corresponding to 4 document corners
document_corners = (top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner)
src = np.array(document_corners, dtype = np.float32)
# define the matrix of target points corresponding to 4 image corners
img_corners = ((0, 0), (0, columns - 1), (rows - 1, 0), (rows - 1, columns - 1))
dst = np.array(img_corners, dtype = np.float32)

# compute the affine transform matrix that relates two images (requires coordinates of 3 points from both)
# src = np.float32([[x[1], x[0]] for x in src])
# dst = np.float32([[x[1], x[0]] for x in dst])
M = cv2.getAffineTransform(src[:3], dst[:3])
rectified = cv2.warpAffine(img, M, (columns, rows))
show_two_imgs(img, rectified, 'Original', 'After Affine transform')
