import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from Refinement.procrustes_analysis import my_procrustes
from Tools import config as cfg


def run(model=cfg.bone_structures[0], new_shape=None, scale='0_12'):
    # load the model
    trained_model = None
    for bones, i in zip(cfg.bone_structures, range(len(cfg.bone_structures))):
        if model == bones:
            trained_model = cfg.models[i]

    file = open(trained_model, 'rb')
    pca = pickle.load(file)
    # print(pca.mean_.shape)

    # x_ = np.reshape(pca.mean_, (138, 1))  # 138x1
    x_ = pca.mean_  # (138,)

    P = pca.components_.T  # 138xt

    # b = np.reshape(pca.explained_variance_, (10, 1))  # tx1
    b = pca.explained_variance_
    b_max = 3 * np.sqrt(b)
    b_min = -3 * np.sqrt(b)
    # print('B_MAX', b_max)
    # print('B_MIN', b_min)
    b[::] = 0  # initialization
    # print('b',b)
    b_copy = b[::]

    for i in range(10):
        # generating model points
        x = x_ + P.dot(b)

        # aligning points
        centroid = np.mean(new_shape, 0)  # translation
        x = np.reshape(x, (len(x) // 2, 2))

        ref_shape, aligned_shape, dist = my_procrustes(x, new_shape, scaled=False)

        # project y
        y = aligned_shape.flatten()
        y_prime = y / y.dot(x_)

        # update b
        b = P.T.dot(y_prime - x_)
        # print('new b',b)
        # print('b_value 1',b[0])

        if np.array_equal(b, b_copy) or i == 9:
            # restrictions for b
            b = restrict_b(b, b_max, b_min)

            x = x_ + P.dot(b)
            ref, aligned, dist = my_procrustes(new_shape, x, scaled=False)

            mean = get_mean_shape(new_shape, x, x_)
            mean = mean + centroid
            m_x, m_y = mean.T

            plt.plot(m_x, -m_y, 'g-', label='Mean Shape')

            aligned = aligned + centroid
            alx, aly = aligned.T
            plt.plot(alx, -aly, 'r-', label='Fitted Shape')
            break
        b_copy = b[::]

    ## original
    nx, ny = new_shape.T
    plt.plot(nx, -ny, 'b.', label='Landmarks')
    plt.legend(loc=0, fontsize='small')
    plt.axis('off')
    output_path = os.path.join(cfg.resultsFolderPath, model)

    plt.savefig(output_path + '/ASM_' + scale + '.' + cfg.img_format, format=cfg.img_format, bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi)
    plt.close()

    return aligned


def get_params_procrustes(reference, shape, scaled=False):
    mtx1 = np.array(reference, dtype=np.double, copy=True)
    mtx2 = np.array(shape, dtype=np.double, copy=True)

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    if scaled:
        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 = mtx1 / norm1
        mtx2 = mtx2 / norm2

    reference_shape = mtx1.flatten()
    shape = mtx2.flatten()

    a = np.dot(shape, reference_shape) / np.linalg.norm(reference_shape) ** 2

    # separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]

    x = shape[::2]
    y = shape[1::2]

    b = np.sum(x * ref_y - ref_x * y) / np.linalg.norm(reference_shape) ** 2

    scale = np.sqrt(a ** 2 + b ** 2)
    theta = np.arctan(b / max(a, 10 ** -10))  # avoid dividing by 0

    mtx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return scale, mtx


def get_mean_shape(new_shape, x, x_):
    scale, mtx = get_params_procrustes(new_shape, x)
    mean = x_ / scale
    mean = mean.reshape((-1, 2)).T
    mean = np.dot(mtx, mean)
    mean = mean.T

    return mean


def restrict_b(b_last, b_max, b_min):
    for i in range(len(b_last)):
        if b_last[i] > b_max[i]:
            b_last[i] = b_max[i]
        elif b_last[i] < b_min[i]:
            b_last[i] = b_min[i]
    return b_last


if __name__ == '__main__':
    new_shape = np.array(
        [(155.0, 322.0), (140.0, 525.5), (152.5, 520.0), (153.0, 514.0), (163.5, 507.5), (158.5, 500.5), (152.5, 493.5),
         (156.5, 487.5), (152.5, 481.5), (147.33333333333334, 476.0), (156.5, 466.5), (150.0, 455.5), (159.5, 445.5),
         (155.0, 435.0), (163.0, 425.0), (169.0, 410.5), (176.0, 400.5), (187.0, 402.5), (199.5, 395.0), (207.5, 383.5),
         (215.5, 384.5), (219.5, 382.0), (226.0, 374.0), (229.0, 374.0), (230.5, 358.0), (231.0, 353.5), (230.5, 342.5),
         (226.5, 333.5), (224.0, 311.0), (217.5, 312.0), (205.0, 306.0), (198.0, 301.0), (188.0, 298.0), (177.5, 297.0),
         (169.0, 293.0), (157.5, 305.5), (149.5, 310.5), (142.0, 315.0), (135.0, 311.0), (129.5, 317.0), (128.0, 339.5),
         (129.0, 338.5), (127.5, 352.5), (122.5, 360.5), (113.0, 365.5), (101.5, 370.5), (83.0, 371.0), (83.0, 365.5),
         (70.0, 360.5), (57.0, 356.0), (47.0, 359.5), (40.0, 367.0), (37.0, 382.5), (30.5, 392.5), (30.0, 405.5),
         (28.0, 417.0), (22.5, 427.5), (24.5, 439.0), (27.0, 451.0), (33.5, 459.5), (40.5, 464.0), (46.0, 470.5),
         (50.5, 480.0), (54.5, 492.5), (54.0, 498.5), (60.5, 509.5), (62.0, 513.5), (68.5, 523.5), (65.5, 532.5)])
    run(new_shape=new_shape)
