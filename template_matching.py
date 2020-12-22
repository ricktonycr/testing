import numpy as np
import matplotlib.pyplot as plt

from skimage import data, io
from skimage.measure import regionprops
from skimage.morphology import label
from skimage.feature import match_template, peak_local_max
from Tools import config as cfg
from scipy.ndimage.filters import gaussian_gradient_magnitude

#image = data.coins()
#coin = image[170:220, 75:130]

image = io.imread(cfg.imagePath, as_grey=True)
patch = image[400:500, 217:317].copy()

image = image[324:824, 87:387]

#image = gaussian_gradient_magnitude(image, sigma=2)
#patch = gaussian_gradient_magnitude(patch, sigma=2)



print('image shape', image.shape)
print('patch shape', patch.shape)

# result = match_template(image, patch, pad_input=True, mode='constant', constant_values=(0,0),)
result = match_template(image, patch)
thresh = 0.7

res = result > thresh
c = label(res, background=0)
reprop = regionprops(c)
print('number', len(reprop))
print('result shape', result.shape)
print('MIN - MAX', np.min(result), np.max(result))
ij = np.unravel_index(np.argmax(result), result.shape)
print(type(ij))
x, y = ij[::-1]
x += patch.shape[0] / 2
y += patch.shape[1] / 2

print(x,y)

plt.imshow(np.clip(result, 0,1), cmap='gray')
plt.show()


fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2, adjustable='box-forced')

ax1.imshow(patch, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(image, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')
# highlight matched region
hpatch, wpatch = patch.shape
rect = plt.Rectangle((x-wpatch/2, y-hpatch/2), wpatch, hpatch, edgecolor='r', facecolor='none')
ax2.add_patch(rect)

ax3.imshow(result, cmap=plt.cm.gray)
ax3.set_axis_off()
ax3.set_title('match_template\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()
