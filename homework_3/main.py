# Unsharp Masking technique (USM)
# technique to improve the sharpness of an image
# sharpened = original + (original − blurred) × amount
import cv2
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]


def show_img(img):
    plt.imshow(img)
    plt.show()


def show_two_imgs(img1, img2):
    plt.subplot(121), plt.imshow(img1)
    plt.subplot(122), plt.imshow(img2)
    plt.show()


img = cv2.imread('resources/flowers.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255
show_img(img)

# blur original image removing high frequency elements
blurred = cv2.GaussianBlur(img, ksize = (9, 9), sigmaX = 5)
show_two_imgs(img, blurred)

# delete everything from original image, leave only sharp edges
diff = img - blurred
diff[diff < 0] = 0
diff[diff > 1] = 1
show_two_imgs(img, diff)

# apply USM
for amount in range(0, 10):
    print(amount)
    sharpened = img + diff * amount
    sharpened[sharpened < 0] = 0
    sharpened[sharpened > 1] = 1
    show_two_imgs(img, sharpened)

# Answers:
# - the reasonable value for 'amount' is up to 3 (at least for this picture)
# - if it's too small, the outcome picture does not differ from the source one much and edges don't look sharper
# - if it's too large, the edge areas reach color ceiling (255) and the outcome image looks over saturated
