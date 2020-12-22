import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import gaussian_gradient_magnitude
from skimage import io

from Tools import config as cfg, pyramid_gaussian

"""
x, y = femur_shape.T

img_gradient = gaussian_gradient_magnitude(img, 1.5)

plt.imshow(img, cmap=plt.cm.gray)
fig, ax = plt.subplots()
ax.imshow(img_gradient, cmap=plt.cm.gray)
ax.plot(x,y, '.r', ms=2)

for point in femur_shape:
    ax.add_patch(Rectangle(point-10, 20, 20, fill= False, edgecolor='green'))

plt.show()

"""


def get_gaussian_gradient_magnitude(image, sigma=cfg.sigma):
    image_gradient = gaussian_gradient_magnitude(image, sigma)
    return image_gradient


def show_gradient_image(image):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Image Gradient')
    plt.axis('off')
    plt.show()
    return image


def show_gradient_image_patches(image_gradient, shape, ps):
    # p = cfg.training_patch_shape[0]
    p = ps
    sx, sy = shape.T
    fig, ax = plt.subplots()
    ax.imshow(image_gradient, cmap=plt.cm.gray)
    ax.plot(sx, sy, '.r', ms=2)
    ax.set_title('Patch Size = ' + str(p))
    ax.axis('off')
    for point in shape:
        ax.add_patch(Rectangle(point - p / 2, p, p, fill=False, edgecolor='g'))
    plt.show()


def sample_patches(image, shape, patch_size):
    image = np.pad(image, cfg.padding, 'constant', constant_values=(0, 0))
    _shape = shape + cfg.padding
    patches = []

    # patches = [crop_image(image, point[0], point[1], patch_size) for point in _shape]

    for point in _shape:
        patch = crop_image(image, point[0], point[1],
                           size=patch_size)  # se debe mandar la dimension del patch de acuerdo a la escala
        patches.append(patch)
    # added
    # show_gradient_image_patches(image,_shape, patch_size[0])
    return patches


def crop_image(image, x_coord, y_coord, size):
    x = int(round(x_coord))
    y = int(round(y_coord))
    x_limit = size[0] // 2
    y_limit = size[1] // 2
    cropped_image = image[y - y_limit:y + y_limit,
                    x - x_limit:x + x_limit].copy()  ## Posible modificación al momento de redondeo
    return cropped_image


if __name__ == '__main__':
    path = '/home/ggutierrez/RayosX/H1.JPG'
    img = io.imread(path, as_grey=True)

    femur_shape = np.array(
        [(141.33333333333334, 610.6666666666666), (148.0, 604.0), (155.33333333333334, 598.6666666666666),
         (163.33333333333334, 592.0), (169.33333333333334, 584.0), (172.66666666666666, 575.3333333333334),
         (174.66666666666666, 566.6666666666666), (173.5, 556.5), (168.66666666666666, 548.0),
         (165.33333333333334, 540.0), (168.0, 533.0), (172.0, 527.5), (176.0, 521.5), (181.5, 515.5), (186.5, 509.5),
         (190.5, 504.0), (196.5, 499.5), (202.5, 495.0), (209.0, 492.0), (219.0, 493.5), (231.0, 495.0), (244.0, 495.5),
         (254.0, 489.0), (262.5, 480.0), (268.5, 471.5), (274.0, 462.5), (275.5, 451.0), (275.5, 438.0), (274.0, 418.5),
         (273.0, 406.5), (267.5, 396.5), (259.5, 387.5), (249.0, 381.0), (238.0, 376.5), (226.0, 373.5), (214.0, 372.5),
         (200.5, 374.5), (187.5, 378.5), (176.5, 384.0), (166.0, 392.5), (160.0, 402.5), (155.5, 414.0), (149.5, 422.5),
         (138.5, 425.0), (126.0, 425.0), (115.5, 424.5), (105.5, 420.0), (99.5, 414.0), (91.5, 405.5), (81.0, 400.0),
         (70.5, 399.5), (61.5, 410.0), (51.0, 420.5), (43.5, 431.0), (38.5, 443.5), (31.0, 456.0), (26.0, 468.0),
         (22.5, 483.0), (25.5, 496.5), (32.5, 507.0), (37.5, 517.5), (41.5, 528.5), (44.5, 540.5), (46.5, 552.0),
         (48.0, 563.0), (49.0, 574.5), (50.0, 585.0), (51.0, 598.0), (51.0, 609.0)])
    pelvis_shape = np.array(
        [(121.0, 302.0), (129.0, 304.0), (136.5, 307.5), (142.5, 313.5), (145.5, 321.5), (149.5, 328.0), (155.5, 335.0),
         (158.0, 343.0), (159.5, 352.0), (162.5, 361.5), (163.5, 370.5), (169.6, 369.2), (176.4, 368.8), (184.8, 366.4),
         (192.4, 365.2), (199.6, 364.4), (206.8, 363.6), (214.4, 363.2), (222.4, 363.6), (230.4, 365.6), (238.0, 367.2),
         (245.6, 369.6), (250.8, 373.6), (256.8, 377.2), (262.4, 380.8), (268.4, 384.8), (274.0, 388.8), (278.8, 393.6),
         (282.0, 398.0), (286.0, 403.2), (289.6, 408.4), (292.4, 414.4), (295.6, 420.0), (296.4, 425.6), (296.8, 431.6),
         (296.8, 438.0), (296.4, 443.6), (299.2, 449.6), (303.6, 455.6), (308.8, 461.2), (308.8, 467.2), (263.2, 480.4),
         (262.5, 491.5), (261.0, 505.0), (264.0, 519.0), (267.0, 530.5), (271.5, 542.0), (278.5, 550.0), (288.0, 556.0),
         (298.5, 563.5), (311.0, 572.5), (323.0, 578.5), (338.0, 580.5), (352.0, 584.5), (365.5, 587.5), (378.0, 584.0),
         (386.5, 572.0), (398.0, 561.0), (405.0, 550.5), (415.0, 541.0), (429.0, 536.5), (445.0, 530.0), (447.0, 514.5),
         (448.5, 498.0), (449.5, 481.5), (447.0, 467.5), (436.5, 468.5), (425.0, 468.0), (415.0, 467.5), (403.5, 466.0),
         (393.5, 463.5), (383.5, 459.0), (374.5, 454.0), (365.5, 448.5), (358.0, 441.0), (352.0, 434.0), (346.0, 426.0),
         (340.0, 420.5), (335.5, 412.5), (329.0, 405.5), (324.5, 397.5), (318.5, 389.5), (313.5, 381.5), (308.0, 374.5),
         (303.5, 366.5), (298.5, 358.0), (294.5, 350.5), (289.0, 342.0), (286.5, 333.0), (284.0, 324.5), (282.0, 315.5),
         (282.0, 305.0), (285.5, 295.5), (293.0, 286.0), (295.0, 280.0)])

    image = img
    shape = femur_shape + 50
    images_pyramid = pyramid_gaussian.get_pyramid(image)
    patch_sizes = cfg.patch_sizes

    for img, ps in zip(images_pyramid[::-1], patch_sizes[::-1]):
        img = np.pad(img, cfg.padding, 'constant', constant_values=(0, 0))
        img_grad = get_gaussian_gradient_magnitude(img)
        # show_gradient_image(img_grad)
        # show_gradient_image_patches(img_grad, shape, ps)
        shape = (shape + 50) / cfg.downScaleFactor
    # ax[1].imshow(img_grad, cmap=plt.cm.gray)

    # plt.show()

    # patches = sample_patches(img_grad, shape, (20, 20)) # estos parches se deben almacenar, vec es una matriz de parches
    # plt.imshow(patches[0], cmap=plt.cm.gray)
    # fig, (ax1,ax2) = plt.subplots(1,2, sharex=True, sharey=True)
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Original Image')
    # ax1.axis('off')
    # # print(vec.shape)
    # # Grafico de la imagen y sus parches
    # # ax1.imshow(patches[0])
    # sx, sy = shape.T
    # p = cfg.training_patch_shape[0]
    # # fig, ax = plt.subplots()
    # ax2.imshow(img_grad, cmap=plt.cm.gray)
    # ax2.set_title('Gradient Magnitude using Gaussian Derivative')
    # ax2.axis('off')
    # ax2.plot(sx, sy, '.r', ms=2)
    # for point in shape:
    #     ax2.add_patch(Rectangle(point - p / 2, p, p, fill=False, edgecolor='g'))
    # plt.show()
