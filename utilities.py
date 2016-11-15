import numpy as np
from skimage.color import rgb2hsv
from imread import imread
from scipy import ndimage, random
from skimage.filters import gaussian, sobel_h, sobel_v
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation
from skimage.draw import circle


def discretize(image):
    threshold = 20
    black = image < threshold
    white = image >= threshold
    image[black] = 0
    image[white] = 255
    return image


def edgy(image_name):
    image = imread(image_name, as_grey=True)
    image = gaussian(image, sigma=.97)
    image = custom_sobel(image)
    image = discretize(image)
    return image


def custom_sobel(image):
    edge_horizont = sobel_h(image)
    edge_vertical = sobel_v(image)
    magnitude = np.hypot(edge_horizont, edge_vertical)
    return magnitude


def edgy_color(image_name):
    image = imread(image_name, as_grey=False)

    image = rgb2hsv(image)
    image = image[..., 2]

    objects, objects_count = ndimage.label(image < .5)

    sizes = ndimage.sum(image, objects, range(objects_count + 1))
    mask_size = sizes < 35
    remove_pixel = mask_size[objects]
    objects[remove_pixel] = 0
    labels = np.unique(objects)
    objects = np.searchsorted(labels, objects)

    contours = find_boundaries(objects)
    objects[contours == 0] = 0
    objects = dilation(objects)
    objects = dilation(objects)

    colours = []
    for _ in range(objects_count + 1):
        colours.append([random.randint(256) / 255 for _ in range(3)] + [1])
    colours[0][3] = 0

    mask = np.zeros_like(objects, dtype=np.float64)
    x, y = objects.shape
    mask.resize(x, y, 4)
    for r, row in enumerate(objects):
        for c, element in enumerate(row):
            mask[r, c] = colours[element]

    centroids = find_centroids(objects)
    for c in centroids:
        rr, cc = circle(c[0], c[1], int(x / 100))
        mask[rr, cc] = 1

    return mask


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
