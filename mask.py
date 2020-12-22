import numpy as np
from skimage import io, filters
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from matplotlib import path
from numpy import ma


path_img = '/home/ggutierrez/OADetection/RayosX/H2.JPG'

img = io.imread(path_img, as_grey=True)

print('IMAGE SHAPE', img.shape)


x2 = np.array([(135.0, 431.0), (139.5, 425.5), (143.5, 421.0), (147.0, 416.0), (150.0, 411.5), (152.5, 405.5),
                    (153.5, 400.0), (153.5, 393.5), (150.0, 388.5), (146.0, 383.0), (148.5, 378.5), (150.5, 374.5),
                    (152.0, 370.0), (155.0, 366.0), (158.5, 362.5), (161.0, 358.0), (164.0, 354.5), (169.0, 352.0),
                    (174.5, 351.0), (183.6, 352.4), (192.0, 351.6), (199.6, 346.0), (205.2, 339.6), (209.6, 331.2),
                    (212.0, 322.4), (212.4, 312.0), (210.8, 302.8), (206.8, 293.6), (201.6, 287.6), (198.0, 280.0),
                    (192.0, 275.6), (184.8, 271.2), (175.6, 268.8), (166.0, 268.0), (157.6, 268.4), (149.6, 269.6),
                    (142.0, 273.2), (136.4, 277.6), (132.4, 282.4), (128.4, 287.6), (125.6, 293.6), (125.2, 300.8),
                    (121.2, 306.0), (115.6, 311.2), (106.4, 311.2), (97.6, 309.6), (88.4, 306.0), (82.8, 299.6),
                    (78.8, 292.0), (67.2, 290.4), (56.8, 290.8), (50.4, 296.4), (50.8, 305.6), (47.2, 315.6),
                    (44.8, 324.8),
                    (41.2, 334.8), (37.6, 344.0), (35.2, 354.8), (36.4, 364.8), (41.6, 369.6), (46.8, 375.6),
                    (49.6, 381.2),
                    (52.8, 387.6), (56.8, 394.0), (58.8, 401.2), (61.6, 408.4), (63.2, 416.4), (64.8, 423.6),
                    (64.8, 430.0)])

plt.imshow(img)
x,y = x2.T
plt.plot(x,y, '.-r')

x2_ = x2+8.7
x1,y1 = x2_.T
plt.plot(x1,y1, '.-b')
plt.show()
new = np.append(x2, [x2[0]], axis=0)
new_ = np.append(x2_, [x2_[0]], axis=0)

plt.imshow(img)
x,y = new.T
plt.plot(x,y, '-r')

x2,y2 =new_.T
plt.plot(x2,y2, '-b')
plt.show()

# print(x2.shape)
print(new.shape)


closed_path = path.Path(new)
closed_path_2 = path.Path(new_)

# Get the points that lie within the closed path
idx = np.array([[(i,j) for i in range(img.shape[1])] for j in range(img.shape[0])]).reshape(np.prod(img.shape),2)
mask = closed_path.contains_points(idx).reshape(img.shape)
mask2 = closed_path_2.contains_points(idx).reshape(img.shape)
print(mask.shape)
#plt.imshow(mask)
plt.imshow(mask2)
plt.show()
print('MASK',mask)
# Invert the mask and apply to the image
#mask = np.invert(mask)
masked_data = ma.array(img.copy(), mask=mask)
masked_data2 = ma.array(img.copy(), mask=mask2)
print(masked_data)
plt.imshow(masked_data)
plt.imshow(masked_data2)
plt.title('masked_data')
plt.show()

intersection = np.logical_and(mask, mask2)
union = np.logical_or(mask, mask2)
iou_score = np.sum(intersection) / np.sum(union)
print('SCORE', iou_score)