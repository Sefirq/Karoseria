import numpy as np
import os
from skimage.color import rgb2grey
from imread import imread
from skimage.filters import gaussian, sobel_h, sobel_v
from skimage.data import imread


def list_of_images():
    DATASETS_PATH = os.path.join(os.path.realpath("__file__"), "../imgs_easy")  # sciezka do podfolderu ze zdjeciami
    for root, directory, files in os.walk(os.path.abspath(DATASETS_PATH)):  # dla plikow w folderze o podanej sciezce
        for image in files:
            yield os.path.join(root, image)


def discretize(image):
    threshold = 20
    black = image < threshold
    white = image >= threshold
    image[black] = 0
    image[white] = 255
    return image


def edgy(image_name):
    image = imread(image_name, as_grey=True)
    # image = gaussian(image, sigma=.97)
    # image = custom_sobel(image)
    # image = discretize(image)
    return image


def custom_sobel(image):
    edge_horizont = sobel_h(image)
    edge_vertical = sobel_v(image)
    magnitude = np.hypot(edge_horizont, edge_vertical)
    return magnitude


def edgy_color(image_name):
    image = imread(image_name, as_grey=False)
    image = rgb2grey(image)
    image = gaussian(image, sigma=3)
    image = custom_sobel(image)
    return image


def find_centroids(labels):
    centroids_all = {}
    for i, row in enumerate(labels):
        for j, element in enumerate(row):
            if element not in centroids_all:
                centroids_all[element] = {'x': 0, 'y': 0, 'n': 0}
            centroids_all[element]['x'] += i
            centroids_all[element]['y'] += j
            centroids_all[element]['n'] += 1

    centroids = []
    for key, value in centroids_all.items():
        if key != 0:
            centroids.append((value['x'] / value['n'], value['y'] / value['n']))
    return centroids
