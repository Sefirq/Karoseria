import numpy as np
import os
from PIL import Image
from skimage.color import rgb2grey, rgb2hsv
from imread import imread
from skimage.filters import gaussian, sobel_h, sobel_v, threshold_otsu, scharr
from skimage.data import imread
from skimage.measure import find_contours
from skimage.feature import canny
from skimage.morphology import erosion, square, dilation
from skimage.draw import polygon
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull


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


def edgy_color(image_name, plot_nr):
    png = Image.open(image_name)
    png.load()  # required for png.split()

    try:
        image = Image.new("RGB", png.size, (255, 255, 255))
        image.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    except IndexError:
        image = png

    bw = rgb2hsv(image)  # obrazek biało-czarny
    bw = bw[..., 2]
    avg = bw.mean()
    mini = bw.min()
    for i, r in enumerate(bw):  # by pozbyć się chmur
        for j, px in enumerate(r):
            if px > mini + 0.28:
                bw[i][j] = avg  # uśrednij piksel [i][j]
    bw = bw > threshold_otsu(bw)  # biało-czarna tablica po progowaniu
    bw = gaussian(bw, sigma=5)
    bw = scharr(bw)
    bw = erosion(bw, square(6))
    bw = dilation(bw, square(6))

    points = []
    for i, row in enumerate(bw):
        for j, elem in enumerate(row):
            if elem > 0.01:
                points.append([j, i])
    points = np.array(points)
    hull = ConvexHull(points)
    contour = points[hull.vertices, 0], points[hull.vertices, 1]

    return bw, contour
    return bw, ([], [])


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
