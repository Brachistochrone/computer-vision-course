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

# define the matrix of source points corresponding to 4 document corners (row x column)
document_corners = (top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner)
src = np.array(document_corners, dtype = np.float32)
# define the matrix of target points corresponding to 4 image corners (column x row)
img_corners = ((0, 0), (columns - 1, 0), (0, rows - 1), (columns - 1, rows - 1))
dst = np.array(img_corners, dtype = np.float32)

# Document Rectification

# Affine transform (does not help since it guarantees parallelism - not parallel lines stay not parallel)
# compute the affine transform matrix that relates two images (requires coordinates of 3 points from both images)
# using first 3 points
M = cv2.getAffineTransform(src[:3], dst[:3])
rectified = cv2.warpAffine(img, M, (columns, rows))
show_two_imgs(img, rectified, 'Original', 'After Affine transform (first 3 points)')
# using last 3 points
M = cv2.getAffineTransform(src[1:4], dst[1:4])
rectified = cv2.warpAffine(img, M, (columns, rows))
show_two_imgs(img, rectified, 'Original', 'After Affine transform (last 3 points)')
# using all 4 points
M, inliers = cv2.estimateAffine2D(src, dst)
rectified = cv2.warpAffine(img, M, (columns, rows))
show_two_imgs(img, rectified, 'Original', 'After Affine transform (all 4 points)')

# Homography estimation
# compute homography matrix
M = cv2.getPerspectiveTransform(src, dst)
rectified = cv2.warpPerspective(img, M, (columns, rows))
show_two_imgs(img, rectified, 'Original', 'After Homography')

# Answers:
# - affine transform does not work in this example because it guarantees parallelism
#   (when parallel lines stay parallel after transform and vice versa), and since vertical edges of our document are
#   not parallel, they remain not parallel and don't fit in the frame
# - the values of inliers are [[1], [1], [0], [1]] that corresponds to the corners of the document that coincide
#   with the corners of the frame (0 - represents the corner that lies outside of the picture)
# - the result after applying homography looks better because all four corners of the document fit into the picture
#   since homography does not have to stick to parallelism and can change the perspective of the image
