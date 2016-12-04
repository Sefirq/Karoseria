import os

import numpy as np
from pandas import read_csv
from PIL import Image
from scipy.spatial import ConvexHull
from skimage.color import rgb2hsv
from skimage.filters import gaussian, threshold_otsu, scharr
from skimage.morphology import erosion, square, dilation
from skimage.draw import polygon_perimeter, polygon
from skimage.measure import moments, moments_central, moments_normalized, moments_hu
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus


def list_of_images(number=-1):
    i = 0
    data_sets_path = os.path.join(os.path.realpath("__file__"), "../imgs_easy")  # sciezka do podfolderu ze zdjeciami
    for root, directory, files in os.walk(os.path.abspath(data_sets_path)):  # dla plikow w folderze o podanej sciezce
        for image in files:
            if i == number:
                return
            yield i, os.path.join(root, image)
            i += 1


def set_of_images(number=-1):
    setofimg = set()
    i = 0
    data_sets_path = os.path.join(os.path.realpath("__file__"), "../imgs_easy")  # sciezka do podfolderu ze zdjeciami
    for root, directory, files in os.walk(os.path.abspath(data_sets_path)):  # dla plikow w folderze o podanej sciezce
        for image in files:
            if i == number:
                return setofimg
            setofimg.add(os.path.join(root, image))
            i += 1
    return setofimg


def discretize(image):
    threshold = 20
    black = image < threshold
    white = image >= threshold
    image[black] = 0
    image[white] = 255
    return image


def moments_of_image(polygon_image):
    m = moments(polygon_image)
    cm = moments_central(polygon_image, m[0, 1] / m[0, 0], m[1, 0] / m[0, 0])
    nm = moments_normalized(cm)
    hm = {}
    for i, m in enumerate(moments_hu(nm)):
        hm['hu{}'.format(i)] = m
    return hm


def feature_detection(shape, cont):
    height_of_car = (max(cont[1]) - min(cont[1])) / shape[0]
    height_divided_by_width = (max(cont[1]) - min(cont[1])) / (max(cont[0]) - min(cont[0]))
    polygon_perimeter_array = np.zeros(shape)
    polygon_array = np.zeros(shape)
    rr, cc = polygon_perimeter(cont[0], cont[1])
    rr2, cc2 = polygon(cont[0], cont[1])
    polygon_perimeter_array[cc, rr] = 1  # array with 1's on the perimeter of contour
    polygon_array[cc2, rr2] = 1  # array with 1's on the whole polygon, bounded by contour
    perimeter = polygon_perimeter_array.sum()
    area = polygon_array.sum()
    how_much_of_picture = area / (shape[0] * shape[1])  # ratio of car area to whole picture area
    hu_moments_of_image = moments_of_image(polygon_array)
    features = {'PpA': perimeter / area, 'H': height_of_car, 'cov': how_much_of_picture, 'HpW': height_divided_by_width}
    features.update(hu_moments_of_image)
    return features


def edgy_color(image_name, class_of_image):
    png = Image.open(image_name)
    png.load()  # required for png.split()

    try:
        image = Image.new("RGB", png.size, (255, 0, 255))
        image.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    except IndexError:
        image = png
    bw = rgb2hsv(image)  # obrazek biało-czarny
    x, y, _ = bw.shape
    b = 2
    bw = bw[b:x - b, b:y - b, 2]
    del png, image
    avg = bw.mean()
    mini = bw.min()
    for i, r in enumerate(bw):  # by pozbyć się tła
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
    features_of_image = feature_detection(bw.shape, contour)
    image_description = {'class': class_of_image}
    image_description.update(features_of_image)
    image_description['NoV'] = len(vertices)
    # print(image_description)
    # above f_o_i is a list of circum/area, height, area/area_of_image and all the Hu moments for this picture
    # probably the first one is not a good feature - needs further analysis
    return contour, image_description


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


def to_csv(filename, results):
    file = open(filename, 'w')
    header = []
    for label, _ in results[0].items():
        header.append(label)

    file.write(','.join(header) + '\n')

    for row in results:
        row_ = []
        for label in header:
            row_.append(str(row[label]))
        file.write(','.join(row_) + '\n')
    file.close()
