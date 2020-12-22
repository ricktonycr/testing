import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import data
from skimage.transform import pyramid_gaussian

from Tools import config as cfg


def run():
    imagePath = '/home/ggutierrez/RayosX/H3.JPG'
    image = data.load(imagePath)
    rows, cols, dim = image.shape
    pyramid = tuple(pyramid_gaussian(image, max_layer=3, sigma=2, downscale=cfg.downScaleFactor))

    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    fig.subplots_adjust(hspace=.1, wspace=.001)
    ax = ax.ravel()

    for i, p in enumerate(pyramid[::-1]):
        print(p.shape)
        ax[i].imshow(p)
        ax[i].add_patch(Rectangle((p.shape[1] / 2 - 20, p.shape[0] / 2 - 20), 40, 40, fill=False, edgecolor='g', lw=2))
        ax[i].tick_params(labelsize=6)
        # plt.show()
    # plt.savefig(cfg.resultsFolderPath + 'pyramid_gaussian.png', format='png', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.show()


def get_pyramid(image):
    pyramid = tuple(pyramid_gaussian(image, max_layer=cfg.maxLayer, downscale=cfg.downScaleFactor, multichannel=False))
    return pyramid[::-1]


if __name__ == '__main__':
    run()
