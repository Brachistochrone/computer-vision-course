import cv2
import matplotlib
import numpy as np
import dlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]


def show_two_imgs(img1, img2, title1 = '', title2 = '', cmap1 = None, cmap2 = None):
    plt.subplot(121), plt.title(title1), plt.imshow(img1, cmap = cmap1)
    plt.subplot(122), plt.title(title2), plt.imshow(img2, cmap = cmap2)
    plt.show()


def get_rect(rect):
    x = rect.left()
    y = rect.top()
    width = rect.right() - x
    height = rect.bottom() - y
    return x, y, width, height


def detect_faces(path, detector):
    # load image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # detect faces
    faces = detector(gray, 1)
    # draw rectangle around each face
    result = np.copy(img)
    for face in faces:
        x, y, width, height = get_rect(face)
        random = np.random.randint(0, 255, 3)
        color = (random[0].item(), random[1].item(), random[2].item())
        cv2.rectangle(result, (x, y), (x + width, y + height), color, 3)
    show_two_imgs(img, result, 'Source image', 'Detected faces')


# load the detector
detector = dlib.get_frontal_face_detector()

detect_faces('resources/girl.jpg', detector)
detect_faces('resources/family.jpg', detector)
detect_faces('resources/beard.jpg', detector)
detect_faces('resources/black.jpg', detector)
detect_faces('resources/concert.jpg', detector)
detect_faces('resources/glasses.jpg', detector)
detect_faces('resources/hat.jpg', detector)
detect_faces('resources/mask.jpg', detector)

# the dlib detector is quite robust because it manages to detect faces with beards, glasses, and face masks
