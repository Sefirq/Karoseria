import os

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from skimage.color import rgb2hsv
from skimage.filters import gaussian, threshold_otsu, scharr
from skimage.morphology import erosion, square, dilation


def list_of_images(number=-1):
    i = 0
    data_sets_path = os.path.join(os.path.realpath("__file__"), "../imgs_easy")  # sciezka do podfolderu ze zdjeciami
    for root, directory, files in os.walk(os.path.abspath(data_sets_path)):  # dla plikow w folderze o podanej sciezce
        for image in files:
            if i == number:
                return
            yield i, os.path.join(root, image)
            i += 1


def discretize(image):
    threshold = 20
    black = image < threshold
    white = image >= threshold
    image[black] = 0
    image[white] = 255
    return image


def edgy_color(image_name):
    png = Image.open(image_name)
    png.load()  # required for png.split()

    try:
        image = Image.new("RGB", png.size, (255, 255, 255))
        image.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    except IndexError:
        image = png

    bw = rgb2hsv(image)  # obrazek biało-czarny
    x, y, _ = bw.shape
    b = 2
    bw = bw[b:x - b, b:y - b, 2]

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
    vertices = hull.vertices.tolist()
    if hull.vertices[0] != hull.vertices[-1]:
        vertices.append(hull.vertices[0])
    contour = points[vertices, 0], points[vertices, 1]

    return bw, contour


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
