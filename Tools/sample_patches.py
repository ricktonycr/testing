import os

import numpy as np
from skimage import io

from Tools import config as cfg


def run():
    datasetImages = os.listdir(cfg.datasetRoot)
    datasetImages = filter(lambda element: '.JPG' in element, datasetImages)

    patches_per_image = []
    patch_centres_per_image = []

    for filename in datasetImages:
        imagePath = cfg.datasetRoot + '/' + filename
        image = io.imread(imagePath)
        patches, patchCentres = create_patches_randomly(image)
        patches_per_image.append(patches)
        patch_centres_per_image.append(patchCentres)

    # Convert list to ndarrays
    patch_centres_per_image = np.array(patch_centres_per_image)

    return patches_per_image, patch_centres_per_image


def create_patches_randomly(image, subshape=[], initialization=False):
    """
    Sample patches randomly around the image
    :param image: image for sample patches
    :param subshape: optional, for sample around subshape
    :param initialization:
    :return: Return the patches and centres positions of these patches
    """
    (patch_width, patch_height) = cfg.patch_shape
    (img_h, img_w) = image.shape[0:2]
    l = len(subshape)

    if l == 0 or initialization == True:
        x_values, y_values = sample_whole_image(img_w, img_h)
    else:
        x_values, y_values = sample_around_subshape(subshape, img_w, img_h)
        # image = np.pad(image, ((cfg.padding,), (cfg.padding,)), 'constant', constant_values=(0,0))
        # image = np.pad(image, ((cfg.padding,), (cfg.padding,)), 'median')

    patches = []
    patch_centres = []

    # added
    # fig, ax = plt.subplots()

    for i in range(cfg.num_of_patches):
        x_pos = x_values[i]
        y_pos = y_values[i]
        # added
        # patch = patches_.Rectangle((x_pos, y_pos), patch_width, patch_height, lw=1, edgecolor='g', facecolor='none')
        # ax.add_patch(patch)

        # Calculate the patch centres
        cx_patch = x_pos + (patch_width // 2)
        cy_patch = y_pos + (patch_height // 2)
        patch_centres.append([cx_patch, cy_patch])
        # Extract the patch (cropping the image)
        patch = crop(image, cfg.patch_shape, x_pos, y_pos)
        # patch = image[y_pos:y_pos+patch_height, x_pos:x_pos+patch_width]
        patches.append(patch)
    # added
    ## If we want to see the points
    # if len(subshape) != 0:
    #    a, b = subshape.T
    #    plt.plot(a, b, '.r')

    ## added
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.title('Patch Sampling')
    # plt.axis('off')
    # plt.savefig('testing.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    patches = np.array(patches)
    patch_centres = np.array(patch_centres).transpose()

    return patches, patch_centres


def crop(image, patch, x_pos, y_pos):
    croppedImage = image[y_pos:y_pos + patch[0], x_pos:x_pos + patch[1]].copy()

    return croppedImage


def sample_whole_image(img_w, img_h):
    (patch_width, patch_height) = cfg.patch_shape  # can be modified
    # x_values = [random.randint(0, img_w - patch_width) for p in range(cfg.num_of_patches)]
    # y_values = [random.randint(0, img_h - patch_height) for p in range(cfg.num_of_patches)]
    # using numpy
    x_values = np.random.randint(0, img_w - patch_width, cfg.num_of_patches)
    y_values = np.random.randint(0, img_h - patch_height, cfg.num_of_patches)

    return x_values, y_values


def sample_around_subshape(subshape, img_w, img_h):
    (patch_width, patch_height) = cfg.patch_shape  # can be modified

    x_centroid, y_centroid = np.mean(subshape, axis=0).astype(dtype=np.int64)

    # Checking limits of the subshape
    x_low = x_centroid - cfg.sample_radius - patch_width // 2
    x_sup = x_centroid + cfg.sample_radius - patch_width // 2
    y_low = y_centroid - cfg.sample_radius - patch_height // 2
    y_sup = y_centroid + cfg.sample_radius - patch_height // 2
    
    # Hacer comprobaciones para los bordes de la imagen, no debe de exceder el tamaño de la imagen
    x_low, x_sup, y_low, y_sup = check_limits(x_low, x_sup, y_low, y_sup, img_w, img_h)

    # x_values = [random.randint(x_low, x_sup) for _ in range(cfg.num_of_patches)]
    # y_values = [random.randint(y_low, y_sup) for _ in range(cfg.num_of_patches)]
    # using numpy randint
    x_values = np.random.randint(x_low, x_sup, cfg.num_of_patches)
    y_values = np.random.randint(y_low, y_sup, cfg.num_of_patches)

    return x_values, y_values


def check_limits(x_low, x_sup, y_low, y_sup, img_w, img_h):
    (patch_width, patch_height) = cfg.patch_shape
    if x_low < 0:
        x_low = 0
    if x_sup + patch_width > img_w:
        x_sup = img_w - patch_width

    if y_low < 0:
        y_low = 0
    if y_sup + patch_height > img_h:
        y_sup = img_h - patch_height

    return x_low, x_sup, y_low, y_sup


if __name__ == '__main__':
    run()
