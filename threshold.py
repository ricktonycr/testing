import numpy as np

from skimage.filters import threshold_adaptive, threshold_triangle, threshold_yen
import matplotlib.pyplot as plt
from skimage import data, color, io


imagePath = '/home/ggutierrez/H7RXfinal.png'
#imagePath = '/home/giancarlos/Descargas/test.jpg'
#image = numpy.asarray(data.load(imagePath))
#print image.shape
#print image.dtype
image = io.imread(imagePath)

print(image)

print(type(image))
print(image[355, 146])
image = data.load(imagePath)
print(image.shape)
print(type(image))

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

print(centeroidnp(image))

fig, ax = plt.subplots(1, 3, figsize=(8, 4))

ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('X-Ray')


red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]

#coord = np.where(np.all(image == (255, 0, 0), axis=-1))
#print zip(coord[0], coord[1])

red = (red == 255) & (green == 0) & (blue == 0)
ax[1].imshow(red, cmap=plt.cm.gray)
ax[1].set_title('Femur')
ax[1].axis('off')
print("RED")
#Coordenadas [Y][X]
print(red[200, 200])
print(red[355, 146])
print(red)

green = (green == 255) & (red == 0) & (blue == 0)
ax[2].imshow(green, cmap=plt.cm.gray)
ax[2].set_title('Hip')
ax[2].axis('off')

"""
blue = (blue == 255) & (red == 0) & (green == 0)
ax[1, 1].imshow(blue, cmap=plt.cm.gray)
ax[1, 1].set_title('blue')
ax[1, 1].axis('off')
"""

plt.show()