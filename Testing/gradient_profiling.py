import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import match_template
from sklearn.neighbors.kde import KernelDensity

from Refinement import gradient_profiling as grad_prof
from Tools import config as cfg
from Tools import data_storage as storage


def run(bone, image, sigma, patch_size, shape, scale, idx_scale):
    img_grad = grad_prof.get_gaussian_gradient_magnitude(image, sigma)

    testing_patches = grad_prof.sample_patches(img_grad, shape, (patch_size, patch_size))

    new_shape = []
    shape_ = []
    all_positions = []
    # new features
    path = '/' + bone + '/scale_' + scale
    training_patches_scale = storage.get_patches_(path)

    patch_size = cfg.patch_sizes[idx_scale]

    for n_patch, testing_patch in enumerate(testing_patches):
        print('loading data for landmark {0} - Scale {1}'.format(n_patch, scale))
        training_patches = []
        for mtx_patches in training_patches_scale:
            patch = mtx_patches[:, n_patch]
            patch = np.reshape(patch, (patch_size, patch_size))
            training_patches.append(patch)

        voted_pos = []
        values = []

        ## added
        votes = np.zeros(testing_patch.shape)

        for training_patch in training_patches:
            value, position, votes = get_match_results(training_patch, testing_patch, shape, n_patch, votes)

            values.append(value)
            voted_pos.append(position)

        voted_pos = np.array(voted_pos)

        all_positions.append(voted_pos.T)

        blurred = gaussian_filter(votes, sigma=2)

        ab = np.unravel_index(np.argmax(blurred), blurred.shape)
        vx, vy = ab[::-1]
        vx += training_patches[0].shape[0] / 2
        vy += training_patches[0].shape[1] / 2
        cx, cy = testing_patch.shape[1] / 2, testing_patch.shape[0] / 2
        d_x, d_y = vx - cx, vy - cy  # distance to translate
        pos_x, pos_y = shape[n_patch - 1][0] + d_x, shape[n_patch - 1][1] + d_y  # update landmark position

        shape_.append([pos_x, pos_y])

        print('MAX VOTE POSITION', pos_x, pos_y)

        # new_shape.append([pos_x, pos_y])

        i = np.argmax(values)
        new_shape.append(voted_pos[i])

    all_positions = np.vstack(all_positions)

    # print('all pos', all_positions.shape)

    # shape_kde = density_estimation(all_positions, image)

    votes_x = all_positions[::2, :].ravel()
    votes_y = all_positions[1::2, :].ravel()

    plt.plot(votes_x, votes_y, 'r.', ms=3, markeredgecolor='k', mew=0.5, alpha=0.4)

    shape_ = np.array(shape_)

    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')

    output_path = os.path.join(cfg.resultsFolderPath, bone)
    plt.savefig(output_path + '/gradient_profiling_voting_' + scale + '.' + cfg.img_format, format=cfg.img_format,
                bbox_inches='tight', pad_inches=0, dpi=cfg.dpi)
    plt.close()

    new_shape = np.array(new_shape)
    plot_shape(image, new_shape, shape_, scale, bone)

    return shape_


def get_match_results(training_patch, testing_patch, shape, n_patch, votes):
    # t0 = time.time()
    result = match_template(testing_patch, training_patch, pad_input=False)
    # print('MATCH TEMPLATE TIME', time.time() - t0)
    # print('RESULT SHAPE', result.shape)

    ij = np.unravel_index(np.argmax(result), result.shape)  # coordinates of max element
    x, y = ij[::-1]  # position in the image

    votes[y, x] += result[y, x]

    value = result[y, x]

    x += training_patch.shape[0] / 2
    y += training_patch.shape[1] / 2

    cx, cy = testing_patch.shape[1] / 2, testing_patch.shape[0] / 2
    d_x, d_y = x - cx, y - cy  # distance to translate
    pos_x, pos_y = shape[n_patch - 1][0] + d_x, shape[n_patch - 1][1] + d_y  # update landmark position

    position = [pos_x, pos_y]

    return value, position, votes


def update_position(testing_patch, n_patch, bone, scale, shape, sigma):
    print('Processing Landmark', str(n_patch))

    path = '/' + bone + '/scale_' + scale
    training_patches = storage.get_patches(path, n_patch)

    print(len(training_patches))

    voted_pos = []
    values = []

    ## added
    votes = np.zeros(testing_patch.shape)
    print(votes.shape)

    for training_patch in training_patches:
        print(multiprocessing.active_children())
        # print('2nd loop')
        result = match_template(testing_patch, training_patch, pad_input=True)

        ij = np.unravel_index(np.argmax(result), result.shape)  # coordinates of max element
        x, y = ij[::-1]  # position in the image
        # added
        # votes[x, y] += 1
        votes[y, x] += result[y, x]
        print('RESULT', n_patch, result[y, x])

        # plot_template_matching(testing_patch,training_patch, result, ij)

        values.append(result[y, x])

        cx, cy = testing_patch.shape[1] / 2, testing_patch.shape[0] / 2
        d_x, d_y = x - cx, y - cy  # distance to translate
        pos_x, pos_y = shape[n_patch - 1][0] + d_x, shape[n_patch - 1][1] + d_y  # update landmark position
        # print('Real pos {0}'.format((pos_x, pos_y)))
        plt.plot(pos_x, pos_y, 'r.', ms=3, markeredgecolor='k', mew=0.5)
        voted_pos.append([pos_x, pos_y])
        # voted_pos = [pos_y, pos_y]
    print('values length', len(values))

    blurred = gaussian_filter(votes, sigma=sigma + 2)
    # plt.imshow(blurred)
    # plt.show()

    ab = np.unravel_index(np.argmax(blurred), blurred.shape)
    vx, vy = ab[::-1]
    d_x, d_y = vx - cx, vy - cy  # distance to translate
    pos_x, pos_y = shape[n_patch - 1][0] + d_x, shape[n_patch - 1][1] + d_y  # update landmark position
    # print('X={0}, Y={1}'.format(pos_x, pos_y))

    # plt.imshow(image, cmap=plt.cm.gray)
    plt.plot(pos_x, pos_y, 'b.', ms=6, markeredgecolor='k')
    new_pos = [pos_x, pos_y]
    # shape_.append([pos_x, pos_y])
    # plt.show()

    print('MAX VOTE POSITION', pos_x, pos_y)

    # new_shape.append([pos_x, pos_y])

    # print(values)
    # print(voted_pos)
    i = np.argmax(values)
    new_pos_ = voted_pos[i]
    # new_shape.append(voted_pos[i])

    return new_pos, new_pos_


def plot_shape(image, shape, shape2, scale, bone):
    x, y = shape.T
    a, b = shape2.T

    plt.imshow(image, cmap=plt.cm.gray)
    plt.plot(x, y, 'ro-', ms=2, markeredgecolor='k', markerfacecolor='r', mew=0.5, lw=0.9, label='Strategy 1')
    plt.plot(a, b, 'o-', color='C0', ms=2, markeredgecolor='k', markerfacecolor='C0', mew=0.5, lw=0.9,
             label='Strategy 2')
    plt.axis('off')
    plt.legend(fontsize='small')
    output_path = os.path.join(cfg.resultsFolderPath, bone)
    plt.savefig(output_path + '/final_gradient_prof_shape_' + scale + '.' + cfg.img_format, format=cfg.img_format,
                bbox_inches='tight', pad_inches=0, dpi=cfg.dpi)
    plt.close()


def plot_template_matching(testing_patch, training_patch, result, position):
    x, y = position[::-1]
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2, adjustable='box-forced')

    ax1.imshow(training_patch, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('Training patch')

    ax2.imshow(testing_patch, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('Testing patch')
    ax2.plot(testing_patch.shape[1] / 2, testing_patch.shape[0] / 2, 'r.', ms=8, mec='k', label='Original landmark')
    # highlight matched region
    hpatch, wpatch = training_patch.shape
    rect = plt.Rectangle((x - wpatch / 2, y - hpatch / 2), wpatch, hpatch, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    ax2.plot(x, y, 'g.', ms=8, mec='k', label='Updated Landmark')
    ax2.legend(loc=0, fontsize='xx-small')

    ax3.imshow(result, cmap=plt.cm.gray)
    ax3.set_axis_off()
    ax3.set_title('match_template\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    print('max value: {0} position: {1}'.format(result[y, x], (x, y)))
    plt.show()


def density_estimation(data, img):
    # img = io.imread(cfg.imagePath, as_grey=True)
    print(data.shape)
    # data = data * 8
    l = data.shape[0] // 2

    x = data[::2, :].ravel()
    y = data[1::2, :].ravel()
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])

    kernel = stats.gaussian_kde(values, bw_method=0.2)
    Z = np.reshape(kernel(positions), X.shape)

    fig = plt.figure()
    # fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    ax.plot(x, y, '+k', markersize=0.5)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.gca().invert_yaxis()
    ax.axis('off')
    plt.show()

    fig, ax_ = plt.subplots()
    ax_.imshow(img, cmap=plt.cm.gray)
    # ax_.set_title('KDE')

    # ax_.pcolormesh(X, Y, Z, shading='goudaud', alpha=0.4, cmap=plt.cm.gist_earth_r)
    ax_.contourf(X, Y, Z, alpha=0.45, cmap=plt.cm.gist_earth_r)

    kde_ = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data.T)
    sc = kde_.score_samples(data.T)

    max = np.argmax(np.exp(sc))
    shape = data[:, max]
    shape = np.reshape(shape, (l, 2))
    # shape *= 8
    # show_landmarks_detected(shape)
    a = shape[:, 0]
    b = shape[:, 1]
    # plt.plot(a,b, 'r.')
    # plt.show()
    # plt.imshow(img, cmap=plt.cm.gray)
    ax_.plot(a, b, 'r.', markersize=8, mec='k', mew=0.3)
    ax_.axis('off')

    # plt.scatter(x, -y, c='k', s=5, edgecolor='white')
    plt.show()
    return shape


if __name__ == '__main__':
    data2 = np.array([(142.0, 532.0), (145.0, 525.5), (150.5, 520.0), (153.0, 514.0), (156.5, 507.5), (158.5, 500.5),
                      (158.5, 493.5), (156.5, 487.5), (152.5, 481.5), (147.33333333333334, 476.0), (151.5, 466.5),
                      (154.0, 455.5), (156.5, 445.5), (159.0, 435.0), (163.0, 425.0), (169.0, 415.5), (176.0, 405.5),
                      (187.0, 397.5), (199.5, 392.0), (207.5, 389.5), (215.5, 386.5), (221.5, 380.0), (226.0, 374.0),
                      (229.0, 366.0), (230.5, 358.0), (231.0, 350.5), (230.5, 342.5), (229.5, 333.5), (222.0, 321.0),
                      (215.5, 312.0), (207.0, 306.0), (198.0, 301.0), (188.0, 298.0), (178.5, 297.0), (167.0, 298.0),
                      (157.5, 300.5), (149.5, 304.5), (142.0, 309.0), (135.0, 316.0), (129.5, 323.0), (128.0, 332.5),
                      (129.0, 342.5), (128.5, 352.5), (122.5, 360.5), (113.0, 367.5), (101.5, 370.5), (88.0, 372.0),
                      (78.0, 365.5), (69.0, 359.5), (57.0, 356.0), (47.0, 359.5), (41.0, 369.0), (37.0, 382.5),
                      (34.5, 394.5),
                      (30.0, 405.5), (26.0, 416.0), (22.0, 427.5), (23.5, 439.0), (27.0, 451.0), (35.5, 459.5),
                      (41.5, 465.0),
                      (46.0, 472.5), (50.5, 480.0), (54.5, 489.5), (57.0, 498.5), (60.5, 506.5), (62.0, 515.5),
                      (64.5, 523.5),
                      (66.5, 532.5)])
