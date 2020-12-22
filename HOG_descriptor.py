import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np

imagePath = '/home/ggutierrez/AP.jpg'
image = color.rgb2gray(data.load(imagePath))

img = image[0:40, 0:40]

fd, hog_image = hog(img, orientations=18, pixels_per_cell=(10, 10),
                    cells_per_block=(4, 4), block_norm='L2-Hys', visualise=True)

fd2, hog_image2 = hog(img, orientations=18, pixels_per_cell=(20, 20),
                    cells_per_block=(2, 2), block_norm='L2-Hys', visualise=True)

f = np.concatenate((fd,fd2))
print(f.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()