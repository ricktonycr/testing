import numpy as np
from skimage.feature import hog

from Tools import config as cfg


def extractHOGFeatures(image):
    feats_1 = hog(image,
                  orientations=cfg.orientations_1,
                  pixels_per_cell=cfg.pixel_per_cell_1,
                  cells_per_block=cfg.cell_per_block_1,
                  block_norm=cfg.block_norm_1,
                  visualize=False)

    feats_2 = hog(image,
                  orientations=cfg.orientations_2,
                  pixels_per_cell=cfg.pixel_per_cell_2,
                  cells_per_block=cfg.cell_per_block_2,
                  block_norm=cfg.block_norm_2,
                  visualize=False)

    feats = np.concatenate((feats_1, feats_2))

    return feats
