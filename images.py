import matplotlib.pyplot as plt
from imread import imread
from utilities import edgy, edgy_color, list_of_images

images = list_of_images()
print(images)

fig = plt.figure(figsize=(60, 40))
fig.set_facecolor('black')
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0,
                    wspace=0.0, hspace=0.0)
for i in range(6):
    print("basic " + str(i + 1) + "/6")
    a = fig.add_subplot(2, 3, i + 1)
    image = edgy(images[i])
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.savefig('basic.pdf', facecolor=fig.get_facecolor())

fig2 = plt.figure(figsize=(60, 60))

for i in range(18):
    print("advanced " + str(i + 1) + "/18")
    original = imread(images[i], as_grey=False)
    b = fig2.add_subplot(7, 3, i + 1)
    image = edgy_color(images[i])
    # plt.imshow(original)
    plt.imshow(image, cmap="Greys_r")
    plt.axis('off')
plt.savefig('advanced.pdf')
