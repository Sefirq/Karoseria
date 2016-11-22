import matplotlib.pyplot as plt
from imread import imread
from utilities import edgy_color, list_of_images

fig = plt.figure(figsize=(60, 60))

for i, image in list_of_images(18):
    print("advanced " + str(i + 1) + "/18... ", end='', flush=True)
    original = imread(image, as_grey=False)
    b = fig.add_subplot(7, 3, i + 1)
    image, (x, y) = edgy_color(image)
    plt.imshow(image, cmap="Greys_r")
    plt.axis('off')
    plt.plot(x, y, 'r--', lw=10)
    print('done')
plt.savefig('advanced.pdf')
