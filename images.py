import matplotlib.pyplot as plt
from imread import imread
from utilities import edgy_color, set_of_images, to_csv
import time

fig = plt.figure(figsize=(60, 60))
classes = list()
results = []
times = []
for i, image in enumerate(set_of_images()):
    classes.append(image[-10:-9])  # classes d for sedan, v for van, s for suv
    print("advanced " + str(i + 1) + "/75... ", end='', flush=True)
    start = time.time()
    (x, y), description = edgy_color(image, classes[i])  # main function
    duration = time.time() - start
    times.append(duration)
    results.append(description)
    print('done')
# print(results)
to_csv('result.csv', results)
print('max {}, min {}, avg {}'.format(max(times), min(times), sum(times)/len(times)))
print('all times')
print(times)
