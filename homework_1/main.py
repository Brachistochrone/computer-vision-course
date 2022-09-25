import cv2
import numpy as np


def display(img_to_show):
    cv2.imshow('IMG', img_to_show)
    cv2.waitKey(0)


# Load an image
img = cv2.imread('resources/aurora.jpg')
display(img)

rows, columns, channels = img.shape
img_rgb = np.zeros_like(img)

# Manual conversion
for row in range(rows):
    for column in range(columns):
        pixel = img[row, column, :]
        img_rgb[row, column, 0] = pixel[2]
        img_rgb[row, column, 1] = pixel[1]
        img_rgb[row, column, 2] = pixel[0]

display(img_rgb)

# Automatic conversion
img_bgr2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display(img_bgr2rgb)

# Split the image into three colour channels
red, green, blue = cv2.split(img)

img1 = cv2.merge([red, green, blue])
img2 = cv2.merge([red, blue, green])
img3 = cv2.merge([green, red, blue])
img4 = cv2.merge([blue, red, green])

# Create collage
out1 = np.hstack([img1, img2])
out2 = np.hstack([img3, img4])
out = np.vstack([out1, out2])
display(out)

# Closing all open windows
cv2.destroyAllWindows()
