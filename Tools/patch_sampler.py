import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from skimage import io

from Tools import config as cfg, json_reader, pyramid_gaussian as pg, sample_patches


def centroid(points):
    x = 0
    y = 0
    n = points.__len__()
    for p in points:
        x += p[0]
        y += p[1]

    X = x // n
    Y = y // n
    return X, Y


path = cfg.path_json + '/H18.json'
subshapes = json_reader.get_all_subshapes(path)
subshapes /= (cfg.downScaleFactor * 4)

centre = []
shape = np.vstack(subshapes[0])
x_val, y_val = shape.T

# subshape = json_reader.get_subshape(path, 12)
subshape = subshapes[0][3]
(cx, cy) = centroid(subshape)
centre.append((cx, cy))

print(shape.shape)
# print('All:',subshapes)
# print('One:', subshape)
print('Centroid:', cx, cy)

image = io.imread(cfg.imagePath, as_grey=True)
image = pg.get_pyramid(image)[0]
img_h, img_w = image.shape
# image = np.pad(image, ((50,), (50,)), 'constant', constant_values=(0,0))
fig, ax = plt.subplots()

################
# mapa de colores
# c = np.arange(1,20)
# cm = plt.get_cmap('rainbow') # pueden haber otras opciones en lugar de hsv,brg,etc
# no_points = len(c)
# ax.set_prop_cycle(cycler('color', [cm(1.*i/(no_points-1)) for i in range(no_points-1)]))
ax.set_prop_cycle(cycler('color', plt.cm.hsv(np.linspace(0.05, 1, 20))))

(patch_width, patch_height) = cfg.patch_shape
# x_values, y_values = sample_patches.sample_around_subshape(centre)
# x_values, y_values = sample_patches.sample_around_subshape(subshape, img_w, img_h)
x_values, y_values = sample_patches.sample_whole_image(image.shape[1], image.shape[0])
c_pos = (x_values[0] + patch_width / 2, y_values[0] + patch_height / 2)

for i in range(cfg.num_of_patches):
    x_pos = x_values[i]
    y_pos = y_values[i]
    patch = patches.Rectangle((x_pos, y_pos), patch_width, patch_height, lw=1, edgecolor='g', facecolor='none')
    ax.plot(x_pos + patch_width / 2, y_pos + patch_height / 2, '.g', markersize=8, label='Patch Center', zorder=3)
    ax.annotate('$(C_x, C_y)$', xy=(x_pos + patch_width / 2, y_pos + patch_height / 2 - 2), fontweight='bold',
                ha='center', va='center')
    ax.annotate('f', xy=(x_pos + 2, y_pos + 3), fontweight='bold', ha='center', va='center')

    ax.add_patch(patch)

"""
for ss in shape:
    x = []
    y = []
    for i in ss:
        x.append(i[0])
        y.append(i[1])
    ax.plot(x, y, 'r|-', lw=0.5, label='Ground Truth')
"""
# plot shape
x_sh, y_sh = shape.T
ax.plot(x_sh, y_sh, 'r.-', lw=0.5, ms=5, label='Ground Truth')

# for p in shape:
#     ax.plot([p[0], c_pos[0]], [p[1], c_pos[1]], 'y-', alpha=0.6, zorder=2)

for ss in subshapes[0]:
    x = []
    y = []
    for i in ss:
        x.append(i[0])
        y.append(i[1])
    # ax.plot(x, y, 'r.', lw=0.5)
x_s = []
y_s = []
for i in subshape:
    x_s.append(i[0])
    y_s.append(i[1])

ax.plot((c_pos[0], subshape[0][0]), (c_pos[1], subshape[0][1]), 'y-')
ax.annotate('d', xy=((subshape[0][0] + c_pos[0]) / 2, (subshape[0][1] + c_pos[1]) / 2), fontweight='bold', ha='center',
            va='center')
ax.plot((c_pos[0], c_pos[0]), (c_pos[1], subshape[0][1]), '-', c='tab:orange')
ax.annotate('$d_y$', xy=(c_pos[0], (c_pos[1] + subshape[0][1]) / 2), fontweight='bold', ha='center', va='center')
ax.plot((c_pos[0], subshape[0][0]), (subshape[0][1], subshape[0][1]), '-', c='tab:orange')
ax.annotate('$d_x$', xy=((subshape[0][0] + c_pos[0]) / 2, subshape[0][1]), fontweight='bold', ha='center', va='center')

# ax.plot(x_s, y_s, 'r.', markersize=8, label='Subshape')
# ax.plot(cx, cy, 'b+', label='Centroid')


ax.axis('off')
# ax.legend()
plt.imshow(image, cmap=plt.cm.gray)
plt.savefig(cfg.resultsFolderPath + 'testing__patch_sampling.pdf', format='pdf', bbox_inches='tight', pad_inches=0,
            dpi=200)
plt.show()
