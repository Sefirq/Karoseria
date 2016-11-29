import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import hsv2rgb
from skimage.color import rgb2hsv

from utilities import list_of_images


def hsv(imagename, channel):
    png = Image.open(imagename)
    png.load()  # required for png.split()

    try:
        img = Image.new("RGB", png.size, (255, 0, 255))
        img.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    except IndexError:
        img = png
    try:
        bw = rgb2hsv(img)
        x, y, _ = bw.shape
        b = 2
        return bw[b:x - b, b:y - b, channel]
    except:
        return img


def rgb(imagename, channel):
    png = Image.open(imagename)
    png.load()  # required for png.split()

    try:
        img = Image.new("RGB", png.size, (255, 0, 255))
        img.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    except IndexError:
        img = png
    try:
        img = rgb2hsv(img)
        img = hsv2rgb(img)
        x, y, _ = img.shape
        b = 2
        return img[b:x - b, b:y - b, channel]
    except:
        return img


fig = plt.figure(figsize=(20, 200))

for i, image_name in list_of_images(75):
    print('Image {}/75... '.format(i + 1), end='', flush=True)
    for j in range(3):
        print('\rImage {}/75... rgb {}/3... '.format(i + 1, j + 1), end='', flush=True)
        b = fig.add_subplot(75, 6, i * 6 + j + 1)
        image = rgb(image_name, j)
        b.imshow(image, cmap="Greys_r")
        b.axis('off')
    for j in range(3):
        print('\rImage {}/75... rgb 3/3... hsv {}/3... '.format(i + 1, j + 1), end='', flush=True)
        b = fig.add_subplot(75, 6, i * 6 + j + 4)
        image = hsv(image_name, j)
        b.imshow(image, cmap="Greys_r")
        b.axis('off')
    print('done')
    plt.savefig('colours.pdf', bbox_inches='tight')
plt.savefig('colours.pdf')
