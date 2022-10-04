import cv2
import math
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]


def show_img(img, cmap = None):
    plt.imshow(img, cmap = cmap)
    plt.show()


def show_two_imgs(img1, img2, cmap = None):
    plt.subplot(121), plt.imshow(img1, cmap = cmap)
    plt.subplot(122), plt.imshow(img2, cmap = cmap)
    plt.show()


def draw_lines(lines, out):
    for line in lines:
        # get offset and angle for line
        rho = line[0]
        theta = line[1]
        # do some math
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # get X, Y coordinates for two dots of line
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        # draw line
        cv2.line(out, pt1, pt2, 255, 1, cv2.LINE_AA)


img = cv2.imread('resources/dashcam.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, None, fx = 0.5, fy = 0.5)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

show_img(gray, 'gray')

# create edge map
edges = cv2.Canny(gray, threshold1 = 100, threshold2 = 400)
show_two_imgs(gray, edges, 'gray')

# remove everything above horizon
edges[0:350] = 0
show_two_imgs(gray, edges, 'gray')

# find lines
lines = cv2.HoughLines(edges, rho = 2, theta = 2 * np.pi / 180, threshold = 200)
# get rid of one dimension
lines = lines[:, 0, :]

# draw found lines
result = np.zeros_like(edges)
draw_lines(lines, result)
show_two_imgs(edges, result, 'gray')

# filter out lines that are approximately horizontal (+/- 20 degrees)
# horizontal lines correspond to theta = 90 degrees
horizontal = (70 * np.pi / 180, 110 * np.pi / 180)
filtered_lines = []
for line in lines:
    theta = line[1]
    if theta < horizontal[0] or theta > horizontal[1]:
        filtered_lines.append(line)

result = np.zeros_like(edges)
draw_lines(filtered_lines, result)
show_two_imgs(edges, result, 'gray')

# apply k-means clustering to group the detected lines
kmeans = KMeans(n_clusters = 4).fit(filtered_lines)
kmeans_lines = kmeans.cluster_centers_
result = np.zeros_like(edges)
draw_lines(kmeans_lines, result)
show_two_imgs(edges, result, 'gray')

# Answers:
# - maybe that lines are infinite and go above the 'horizon' crossing in the middle
# - definitely, because small steps of 'rho' and 'theta' allow us to detect more lines in the image,
#   and with low resolution we could simply miss some important lines
# - threshold is also very important, because it allows us to filter out lines created by small edges and noise,
#   reduce the number of lines, and get only lines which are closest to main edges
