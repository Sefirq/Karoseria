import matplotlib.pyplot as plt
from imread import imread
from utilities import edgy_color, set_of_images, to_csv

fig = plt.figure(figsize=(60, 60))
classes = list()
results = []
for i, image in enumerate(set_of_images()):
    classes.append(image[-10:-9])  # classes d for sedan, v for van, s for suv
    # print(classes)
    print("advanced " + str(i + 1) + "/75... ", end='', flush=True)
    #original = imread(image, as_grey=False)
    #b = fig.add_subplot(7, 3, i + 1)
    print(image[-10:-9], image)
    (x, y), description = edgy_color(image, classes[i])
    #plt.imshow(image, cmap="Greys_r")
    # plt.imshow(polygon_arr, cmap="Greys_r")
    #plt.axis('off')
    #plt.plot(x, y, 'r--', lw=10)
    results.append(description)
    print('done')
# print(results)
to_csv('result.csv', results)
#classify()
#plt.savefig('advanced.pdf')
