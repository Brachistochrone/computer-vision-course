import cv2
import matplotlib
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [10, 5]


def show_img(img, title = '', b_and_w = False):
    cmap = 'gray' if b_and_w else None
    plt.title(title)
    plt.imshow(img, cmap = cmap)
    plt.show()


def find_closest(array, val):
    i = (np.abs(array - val)).argmin()
    return array[i]


def defuse_quant_error(src_img, err, r, c, chs, b_and_w = False):
    for ch in range(chs):
        try:
            if c + 1 > src_img.shape[1]:
                raise IndexError
            if b_and_w:
                src_img[r, c + 1] = src_img[r, c + 1] + err[ch] * 7 / 16
            else:
                src_img[r, c + 1, ch] = src_img[r, c + 1, ch] + err[ch] * 7 / 16
        except IndexError:
            pass
        try:
            if r + 1 > src_img.shape[0] or c - 1 < 0:
                raise IndexError
            if b_and_w:
                src_img[r + 1, c - 1] = src_img[r + 1, c - 1] + err[ch] * 3 / 16
            else:
                src_img[r + 1, c - 1, ch] = src_img[r + 1, c - 1, ch] + err[ch] * 3 / 16
        except IndexError:
            pass
        try:
            if r + 1 > src_img.shape[0]:
                raise IndexError
            if b_and_w:
                src_img[r + 1, c] = src_img[r + 1, c] + err[ch] * 5 / 16
            else:
                src_img[r + 1, c, ch] = src_img[r + 1, c, ch] + err[ch] * 5 / 16
        except IndexError:
            pass
        try:
            if r + 1 > src_img.shape[0] or c + 1 > src_img.shape[1]:
                raise IndexError
            if b_and_w:
                src_img[r + 1, c + 1] = src_img[r + 1, c + 1] + err[ch] * 1 / 16
            else:
                src_img[r + 1, c + 1, ch] = src_img[r + 1, c + 1, ch] + err[ch] * 1 / 16
        except IndexError:
            pass


# change me
blackAndWhiteImg = False
blackWhitePalette = False
is_bonus = False

img = cv2.imread('resources/vegetables.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if blackAndWhiteImg else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
show_img(img, 'Source picture', blackAndWhiteImg)

# Quantization
# reference palette
palette = np.array([[0], [255]]) if blackWhitePalette else np.array([[0], [64], [192], [255]])

if is_bonus:
    kmeans = KMeans(n_clusters = 16).fit(np.reshape(img, (-1, 1)))
    palette = kmeans.cluster_centers_

img = img.astype(np.float32)
rows = img.shape[0]
columns = img.shape[1]
channels = 1 if blackAndWhiteImg else img.shape[2]
quantized = np.zeros_like(img)

# apply quantization
for row in range(rows):
    for column in range(columns):
        pixel = [img[row, column]] if blackAndWhiteImg else img[row, column]
        # find the closest colour from the palette (using absolute value/Euclidean distance)
        new_pixel = [find_closest(palette[:, 0], pixel[ch]) for ch in range(channels)]
        if blackAndWhiteImg:
            quantized[row, column] = new_pixel[0]
        else:
            quantized[row, column, :] = new_pixel

show_img(quantized.astype(np.uint8), 'Normally quantized picture', blackAndWhiteImg)

# compute average quantization error
mse = np.mean((img - quantized) ** 2)
psnr = 10 * np.log10((255 ** 2) / mse)
print('Quantization MSE:', mse)
print('Quantization PSNR:', psnr, 'dB')

# Floyd-Steinberg dithering algorithm
img_tmp = np.copy(img)
dithering = np.zeros_like(img)

# apply quantization with dithering
for row in range(rows):
    for column in range(columns):
        pixel = [img_tmp[row, column]] if blackAndWhiteImg else img_tmp[row, column]
        # find the closest colour from the palette (using absolute value/Euclidean distance)
        new_pixel = [find_closest(palette[:, 0], pixel[ch]) for ch in range(channels)]
        # compute quantization error per channel
        quant_error = np.subtract(pixel, new_pixel)
        # apply FS dithering algorithm
        defuse_quant_error(img_tmp, quant_error, row, column, channels, blackAndWhiteImg)
        if blackAndWhiteImg:
            dithering[row, column] = new_pixel[0]
        else:
            dithering[row, column, :] = new_pixel

show_img(dithering.astype(np.uint8), 'Quantized with dithering picture', blackAndWhiteImg)

mse = np.mean((img - dithering) ** 2)
psnr = 10 * np.log10((255 ** 2) / mse)
print('Dithering MSE:', mse)
print('Dithering PSNR:', psnr, 'dB')

plt.subplot(131), plt.imshow(img.astype(np.uint8), cmap = 'gray' if blackAndWhiteImg else None), plt.title(
    'Source picture')
plt.subplot(132), plt.imshow(quantized.astype(np.uint8), cmap = 'gray' if blackAndWhiteImg else None), plt.title(
    'Quantized picture')
plt.subplot(133), plt.imshow(dithering.astype(np.uint8), cmap = 'gray' if blackAndWhiteImg else None), plt.title(
    'Quantized with dithering picture')
plt.show()

# Answers:
# - dithered image has higher quantization error
# - dithered image looks way better
# - dithering also works for black and white pictures; to check it change 'blackAndWhiteImg' variable to True
# - it's also possible to use only black and white colors as a palette for the process;
#   to check it change 'blackWhitePalette' variable to True
# - for Bonus change 'is_bonus' variable to True
# - if we use 256 colours, the quantization doesn't take place since all colors stay the same and MSE is negligible
