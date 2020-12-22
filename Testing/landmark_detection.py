import os
from time import time
import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from scipy import stats
from scipy.sparse import csgraph
from skimage import io
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.kde import KernelDensity

import extract_features
from Tools import config as cfg, pyramid_gaussian, sample_patches
from Tools import data_storage as storage


def run(imagepath):
    t0 = time()

    image = io.imread(imagepath, as_gray=True)
    pyramid = pyramid_gaussian.get_pyramid(image)

    cfg.num_of_patches = cfg.num_test_patches  # changing the number of patches

    for img, sc_name in zip(pyramid, cfg.scale_names):
        if sc_name == '0_12':
            init_flag = True
            ss = []
            ns = 0

            patches, centres = sample_patches.create_patches_randomly(img,subshape=ss, initialization=init_flag)
            f = extract_features.extractFeaturesForPatches(patches)

            # 0: femur
            # 1: cadera
            # 2: superior
            # 3: inferior
            d_tilde, f_tilde, c_tilde = build_matrices(cfg.bone_structures[3], sc_name, n_subs=ns)

            l = d_tilde.shape[0] // 2  # number of landmarks

            # Obtener los puntos
            f_hat = np.concatenate((f_tilde, f), axis=1)
            c_bar = compute_C_matrix(centres, l)
            c = np.tile(centres, (l, 1))
            d = compute_D_matrix(f_hat, d_tilde, c_bar, l)
            data = d + c
            kde_ = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data.T)
            sc = kde_.score_samples(data.T)
            max = np.argmax(np.exp(sc))
            shape = data[:, max]
            shape = np.reshape(shape, (l, 2))
            a = shape[:, 0]
            b = shape[:, 1]
            a *= 8
            b *= 8
            # fig, ax_ = plt.subplots()
            # ax_.imshow(image, cmap=plt.cm.gray)
            # ax_.plot(a, b, 'r.', markersize=8, mec='k', mew=0.3)
            # ax_.plot(a[5], b[5], 'b.', markersize=8, mec='k', mew=0.3)
            # ax_.plot(a[12], b[12], 'b.', markersize=8, mec='k', mew=0.3)
            # ax_.axis('off')
            # plt.show()
            a /= 4
            b /= 4

        else:
            for count in range(5):
                init_flag = False
                if count == 4:
                    ss = shape[(4*count):(4*count+5),:]
                else:
                    ss = shape[(4*count):(4*count+4),:]
                ns = count
                # if sc_name != '0_25':
                #     ss = shape[0:4,:]
                
                patches, centres = sample_patches.create_patches_randomly(img,subshape=ss, initialization=init_flag)
                f = extract_features.extractFeaturesForPatches(patches)

                # 0: femur
                # 1: cadera
                # 2: superior
                # 3: inferior
                d_tilde, f_tilde, c_tilde = build_matrices(cfg.bone_structures[3], sc_name, n_subs=ns)

                l = d_tilde.shape[0] // 2  # number of landmarks

                # Obtener los puntos
                f_hat = np.concatenate((f_tilde, f), axis=1)
                c_bar = compute_C_matrix(centres, l)
                c = np.tile(centres, (l, 1))
                d = compute_D_matrix(f_hat, d_tilde, c_bar, l)
                data = d + c
                kde_ = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data.T)
                sc = kde_.score_samples(data.T)
                max = np.argmax(np.exp(sc))
                shape1 = data[:, max]
                shape1 = np.reshape(shape1, (l, 2))
                # a = shape1[:, 0]
                # b = shape1[:, 1]
                # if sc_name == '0_25':
                #     a *= 4
                #     b *= 4
                # elif sc_name == '0_5':
                #     a *= 2
                #     b *= 2
                # fig, ax_ = plt.subplots()
                # ax_.imshow(image, cmap=plt.cm.gray)
                # ax_.plot(a, b, 'r.', markersize=8, mec='k', mew=0.3)
                # ax_.axis('off')
                # plt.show()
                # if sc_name == '0_25':
                #     a /= 2
                #     b /= 2
                if count == 4:
                    shape[(4*count):(4*count+5),:] = shape1[0:5,:]*2  
                else:  
                    shape[(4*count):(4*count+4),:] = shape1[0:4,:]*2

            a = shape[:, 0]
            b = shape[:, 1]
            if sc_name == '0_25':
                a = a*2
                b = b*2
            if sc_name == '1':
                a = a/2
                b = b/2
            # fig, ax_ = plt.subplots()
            # ax_.imshow(image, cmap=plt.cm.gray)
            # ax_.plot(a, b, 'r.', markersize=8, mec='k', mew=0.3)
            # ax_.plot(a[5], b[5], 'b.', markersize=8, mec='k', mew=0.3)
            # ax_.plot(a[12], b[12], 'b.', markersize=8, mec='k', mew=0.3)
            # ax_.axis('off')
            # plt.show()

    izquierdaX = np.copy(a)
    izquierdaY = np.copy(b)


    for img, sc_name in zip(pyramid, cfg.scale_names):
        if sc_name == '0_12':
            init_flag = True
            ss = []
            ns = 0

            patches, centres = sample_patches.create_patches_randomly(img,subshape=ss, initialization=init_flag)
            f = extract_features.extractFeaturesForPatches(patches)

            # 0: femur
            # 1: cadera
            # 2: superior
            # 3: inferior
            d_tilde, f_tilde, c_tilde = build_matrices(cfg.bone_structures[1], sc_name, n_subs=ns)

            l = d_tilde.shape[0] // 2  # number of landmarks

            # Obtener los puntos
            f_hat = np.concatenate((f_tilde, f), axis=1)
            c_bar = compute_C_matrix(centres, l)
            c = np.tile(centres, (l, 1))
            d = compute_D_matrix(f_hat, d_tilde, c_bar, l)
            data = d + c
            kde_ = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data.T)
            sc = kde_.score_samples(data.T)
            max = np.argmax(np.exp(sc))
            shape = data[:, max]
            shape = np.reshape(shape, (l, 2))
            a = shape[:, 0]
            b = shape[:, 1]
            a *= 8
            b *= 8
            # fig, ax_ = plt.subplots()
            # ax_.imshow(image, cmap=plt.cm.gray)
            # ax_.plot(a, b, 'r.', markersize=8, mec='k', mew=0.3)
            # ax_.plot(a[5], b[5], 'b.', markersize=8, mec='k', mew=0.3)
            # ax_.axis('off')
            # plt.show()
            a /= 4
            b /= 4

        else:
            for count in range(5):
                init_flag = False
                if count == 4:
                    ss = shape[(4*count):(4*count+5),:]
                else:
                    ss = shape[(4*count):(4*count+4),:]
                ns = count
                # if sc_name != '0_25':
                #     ss = shape[0:4,:]
                
                patches, centres = sample_patches.create_patches_randomly(img,subshape=ss, initialization=init_flag)
                f = extract_features.extractFeaturesForPatches(patches)

                # 0: femur
                # 1: cadera
                # 2: superior
                # 3: inferior
                d_tilde, f_tilde, c_tilde = build_matrices(cfg.bone_structures[1], sc_name, n_subs=ns)

                l = d_tilde.shape[0] // 2  # number of landmarks

                # Obtener los puntos
                f_hat = np.concatenate((f_tilde, f), axis=1)
                c_bar = compute_C_matrix(centres, l)
                c = np.tile(centres, (l, 1))
                d = compute_D_matrix(f_hat, d_tilde, c_bar, l)
                data = d + c
                kde_ = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data.T)
                sc = kde_.score_samples(data.T)
                max = np.argmax(np.exp(sc))
                shape1 = data[:, max]
                shape1 = np.reshape(shape1, (l, 2))
                # a = shape1[:, 0]
                # b = shape1[:, 1]
                # if sc_name == '0_25':
                #     a *= 4
                #     b *= 4
                # elif sc_name == '0_5':
                #     a *= 2
                #     b *= 2
                # fig, ax_ = plt.subplots()
                # ax_.imshow(image, cmap=plt.cm.gray)
                # ax_.plot(a, b, 'r.', markersize=8, mec='k', mew=0.3)
                # ax_.axis('off')
                # plt.show()
                # if sc_name == '0_25':
                #     a /= 2
                #     b /= 2
                if count == 4:
                    shape[(4*count):(4*count+5),:] = shape1[0:5,:]*2  
                else:  
                    shape[(4*count):(4*count+4),:] = shape1[0:4,:]*2

            a = shape[:, 0]
            b = shape[:, 1]
            if sc_name == '0_25':
                a = a*2
                b = b*2
            if sc_name == '1':
                a = a/2
                b = b/2
            # fig, ax_ = plt.subplots()
            # ax_.imshow(image, cmap=plt.cm.gray)
            # ax_.plot(a, b, 'r.', markersize=8, mec='k', mew=0.3)
            # ax_.plot(a[5], b[5], 'b.', markersize=8, mec='k', mew=0.3)
            # ax_.plot(a[12], b[12], 'b.', markersize=8, mec='k', mew=0.3)
            # ax_.axis('off')
            # plt.show()

    derechaX = np.copy(a)
    derechaY = np.copy(b)

    fig, ax_ = plt.subplots()
    ax_.imshow(image, cmap=plt.cm.gray)
    ax_.plot(a, b, 'r.', markersize=5, mec='k', mew=0.3)
    ax_.plot(izquierdaX, izquierdaY, 'r.', markersize=8, mec='k', mew=0.3)
    ax_.plot(derechaX, derechaY, 'r.', markersize=8, mec='k', mew=0.3)


    IT = (izquierdaY[12] - izquierdaY[5]) / (izquierdaX[12] - izquierdaX[5])
    DT = (derechaY[12] - derechaY[5]) / (derechaX[12] - derechaX[5])

    a1 = 2*izquierdaX[5]-izquierdaX[12]
    a2 = 2*izquierdaX[12]-izquierdaX[5]
    b1 = IT*(a1 - izquierdaX[5]) + izquierdaY[5]
    b2 = IT*(a2 - izquierdaX[5]) + izquierdaY[5]


    c1 = 2*derechaX[5]-derechaX[12]
    c2 = 2*derechaX[12]-derechaX[5]
    d1 = DT*(c1 - derechaX[5]) + derechaY[5]
    d2 = DT*(c2 - derechaX[5]) + derechaY[5]



    HT = (derechaY[12] - izquierdaY[12]) / (derechaX[12] - izquierdaX[12])

    e1 = a1
    e2 = c1
    f1 = HT*(e1 - izquierdaX[12]) + izquierdaY[12]
    f2 = HT*(e2 - izquierdaX[12]) + izquierdaY[12]



    g1 = izquierdaX[5]
    g2 = HT*(g1 - izquierdaX[12]) + izquierdaY[12]
    h1 = derechaX[5]
    h2 = HT*(h1 - izquierdaX[12]) + izquierdaY[12]

    ax_.plot([e1,e2], [f1,f2], 'g', markersize=8, mec='k', mew=0.3)
    ax_.plot([c1,c2], [d1,d2], 'g', markersize=8, mec='k', mew=0.3)
    ax_.plot([a1,a2], [b1,b2], 'g', markersize=8, mec='k', mew=0.3)
    ax_.plot(izquierdaX[5], izquierdaY[5], 'b.', markersize=10, mec='k', mew=0.3)
    ax_.plot(izquierdaX[12], izquierdaY[12], 'b.', markersize=10, mec='k', mew=0.3)
    ax_.plot(derechaX[5], derechaY[5], 'b.', markersize=10, mec='k', mew=0.3)
    ax_.plot(derechaX[12], derechaY[12], 'b.', markersize=10, mec='k', mew=0.3)
    ax_.plot([g1,h1], [g2,h2], 'b.', markersize=10, mec='k', mew=0.3)


    nume1 = izquierdaY[5]*(izquierdaX[12]-g1) + izquierdaY[12]*(g1-izquierdaX[5]) + g2*(izquierdaX[5]-izquierdaX[12])
    deno1 = (izquierdaX[5]-izquierdaX[12])*(izquierdaX[12]-g1) + (izquierdaY[5]-izquierdaY[12])*(izquierdaY[12]-g2)
    rati1 = nume1/deno1
    angl1 = math.atan(rati1)
    deg1  = (angl1*180)/math.pi
    if deg1<0:
        deg1 = deg1 + 180
    print(deg1)

    nume2 = derechaY[5]*(derechaX[12]-h1) + derechaY[12]*(h1-derechaX[5]) + h2*(derechaX[5]-derechaX[12])
    deno2 = (derechaX[5]-derechaX[12])*(derechaX[12]-h1) + (derechaY[5]-derechaY[12])*(derechaY[12]-h2)
    rati2 = nume2/deno2
    angl2 = math.atan(rati2)
    deg2  = (angl2*180)/math.pi
    if deg2<0:
        deg2 = deg2 + 180
        deg2 = 180 - deg2
    print(deg2)

    ax_.text(izquierdaX[12]-20, izquierdaY[12] + 20, round(deg1,2),color='yellow')
    ax_.text(derechaX[12]+20, derechaY[12] + 20, round(deg2,2),color='yellow')

    print('####\tNiña\tNiño')

    na = 'N'
    no = 'N'

    #1-2
    if deg1 > 36 or deg2 > 36:
        na = 'L'
    if deg1 > 41.5 or deg2 > 41.5:
        na = 'G'
    if deg1 > 29 or deg2 > 31:
        no = 'L'
    if deg1 > 33 or deg2 > 35:
        no = 'G'
    print('1-2\t' + na + '\t' + no)

    #3-4
    if deg1 > 31.5 or deg2 > 33:
        na = 'L'
    if deg1 > 36.5 or deg2 > 38.5:
        na = 'G'
    if deg1 > 28 or deg2 > 29:
        no = 'L'
    if deg1 > 32.5 or deg2 > 33.5:
        no = 'G'
    print('3-4\t' + na + '\t' + no)

    #5-6
    if deg1 > 27.5 or deg2 > 29.5:
        na = 'L'
    if deg1 > 32 or deg2 > 34:
        na = 'G'
    if deg1 > 24.5 or deg2 > 27:
        no = 'L'
    if deg1 > 29 or deg2 > 31.5:
        no = 'G'
    print('5-6\t' + na + '\t' + no)

    #7-9
    if deg1 > 25.5 or deg2 > 27:
        na = 'L'
    if deg1 > 29.5 or deg2 > 31.5:
        na = 'G'
    if deg1 > 24.5 or deg2 > 25.5:
        no = 'L'
    if deg1 > 29 or deg2 > 29.5:
        no = 'G'
    print('7-9\t' + na + '\t' + no)

    #2a-3a
    if deg1 > 22 or deg2 > 23.5:
        na = 'L'
    if deg1 > 25.5 or deg2 > 27:
        na = 'G'
    if deg1 > 21 or deg2 > 22.5:
        no = 'L'
    if deg1 > 25 or deg2 > 27:
        no = 'G'
    print('2a-3a\t' + na + '\t' + no)

    #3a-5a
    if deg1 > 18 or deg2 > 21:
        na = 'L'
    if deg1 > 25.5 or deg2 > 25.5:
        na = 'G'
    if deg1 > 19 or deg2 > 20:
        no = 'L'
    if deg1 > 23.5 or deg2 > 24:
        no = 'G'
    print('3a-5a\t' + na + '\t' + no)


    ax_.axis('off')
    plt.show()

    



    '''
    l = d_tilde.shape[0] // 2  # number of landmarks

    # Composed matrix
    f_hat = np.concatenate((f_tilde, f), axis=1)

    c_bar = compute_C_matrix(centres, l)
    c = np.tile(centres, (l, 1))

    d = compute_D_matrix(f_hat, d_tilde, c_bar, l)

    positions_ = d + c

    density_estimation(positions_, img,imagepath)

    '''

    '''
    
    # x = np.exp(sc)
    # print(sc.shape)
    # plt.imshow(positions, cmap=plt.cm.gist_earth_r)
    # plt.show()
    x = []
    y = []
    for s in range(d.shape[1]):
        # displacement = d[:,s]
        # center = centres[:,s].T
        # displacement = np.reshape(displacement,(69,2))
        # shape = displacement + center
        shape = np.reshape(positions_[:, s], (l, 2))
        x.append(shape[:, 0])
        y.append(shape[:, 1])
        show_landmarks_detected(shape)


    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

    fig, ax = plt.subplots(1,2)
    x = np.array(x).ravel()
    y = np.array(y).ravel()


    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.plot(x,-y, 'go')
    # plt.show()
    data = np.vstack((x, y))
    d = data.shape[0]
    n = data.shape[1]
    bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))
    print('BANDWIDTH', bw)

    kde_2 = KernelDensity(bandwidth=bw, metric='euclidean',
                          kernel='gaussian', algorithm='ball_tree')
    kde_2.fit(data.T)

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]


    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x,y])
    kernel = stats.gaussian_kde(values)
    #print(kernel.evaluate(positions).)
    z = np.reshape(kernel(positions).T, X.shape)

    print('z', z.shape)

    ax[0].imshow(np.rot90(z), cmap=plt.cm.gist_earth_r, alpha=0.8)
    ax[0].plot(x, y, 'k.', markersize=1)
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])

    #Z = np.full(img.shape)
    #img[x,y] = np.exp(kde_2.score_samples(positions.T))

    Z = np.reshape(np.exp(kde_2.score_samples(positions.T)), X.shape)
    print('Z', Z.shape)
    #Z[mask] = 1
    ax[1].imshow(img, cmap=plt.cm.gray)
    ax[1].set_title('modified')

    #ax.imshow(Z.T,interpolation='nearest', extent=[xmin, xmax, ymin, ymax], cmap=plt.cm.gist_earth_r)
    #ax.axis(aspect='image')
    levels = np.linspace(0, Z.max(), 50)

    #ax.pcolormesh(X, Y, Z, shading='goudaud', alpha=0.4, cmap=plt.cm.gist_earth_r)
    ax[1].contourf(X, Y, Z, alpha=0.4, cmap=plt.cm.gist_earth_r)

    #ax.contourf(X,Y,Z, levels=levels, alpha=0.4, cmap='YlGnBu', extent=[xmin, xmax, ymin, ymax])

    kde_ = KernelDensity(kernel='gaussian', bandwidth=bw).fit(positions_.T)
    sc = kde_.score_samples(positions_.T)

    max = np.argmax(np.exp(sc))
    shape = positions_[:, max]
    shape = np.reshape(shape, (l, 2))
    #shape *= 8
    # show_landmarks_detected(shape)
    a = shape[:, 0]
    b = shape[:, 1]
    # plt.plot(a,b, 'r.')
    # plt.show()
    #plt.imshow(img, cmap=plt.cm.gray)
    ax[1].plot(a, b, 'r.', markersize=2)

    # plt.scatter(x, -y, c='k', s=5, edgecolor='white')
    plt.show()

    """
    kde2 = KernelDensity().fit(data)

    r = np.linspace(0, 100, 150)
    X, Y = np.meshgrid(r, r)

    plot_data = np.vstack((X.ravel(), Y.ravel())).T

    log_dens = kde2.score_samples(plot_data)

    plt.pcolormesh(X, Y, log_dens.reshape(X.shape), cmap='Purples')
    plt.colorbar()
    plt.plot(x, y, 'b.')
    plt.show()

    """

    #nbins = 20
    #plt.title('Hexbin')
    #plt.imshow(img, cmap=plt.cm.gray)
    #plt.hexbin(x, y, cmap=plt.cm.BuGn_r)
    #plt.show()

    """
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    #xx, yy = np.meshgrid(x, y)
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.imshow(img, cmap=plt.cm.gray)
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud')
    #plt.show()
    """
    # Cross Validation
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(positions_.T)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    """
    x_grid = np.linspace(-4.5, 138, 500)
    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(positions.T))

    fig, ax = plt.subplots()
    ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
    ax.hist(positions, 130, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
    ax.legend(loc='upper left')
    ax.set_xlim(-4.5, 138)
    
    
    
    kde_ = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(positions_.T)
    sc = kde_.score_samples(positions_.T)

    max = np.argmax(np.exp(sc))
    shape = positions_[:, max]
    shape = np.reshape(shape, (l, 2))
    # show_landmarks_detected(shape)
    a = shape[:, 0]
    b = shape[:, 1]
    # plt.plot(a,b, 'r.')
    # plt.show()
    plt.imshow(img, cmap=plt.cm.gray)
    plt.plot(a, b, 'r.', markersize=2)
    plt.show()
    """
    print('\ntime elapsed: %.2fs' % (time() - t0))
    '''


def get_estimated_landmarks(bone, image, scale_name, subshape=[], n_subs=0, init_flag=False):
    patches, patch_centres = sample_patches.create_patches_randomly(image, subshape, initialization=init_flag)
    f = extract_features.extractFeaturesForPatches(patches)

    d_tilde, f_tilde, c_tilde = build_matrices(bone, scale_name, n_subs)

    l = d_tilde.shape[0] // 2  # number of landmarks

    # Composed matrix
    f_hat = np.concatenate((f_tilde, f), axis=1)

    c_bar = compute_C_matrix(patch_centres, l)
    c = np.tile(patch_centres, (l, 1))

    d = compute_D_matrix(f_hat, d_tilde, c_bar, l)

    predicted_positions = d + c
    # added
    if scale_name == cfg.init_scale:
        plot_predicted_positions(image, predicted_positions, l, bone)
    # plot_predicted_positions(image, predicted_positions, l)

    estimated_landmarks = kernel_density_estimation(predicted_positions)  # maybe we can send the "l"
    # added
    # plot_estimated_landmarks(image, estimated_landmarks)

    return estimated_landmarks


def density_estimation(data, img,imagepath):
    img = io.imread(imagepath, as_gray=True)
    print(data.shape)
    data = data * 4
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
    #ax_.contourf(X, Y, Z, alpha=0.45, cmap=plt.cm.gist_earth_r)

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
    print(a[0])
    print(b[0])
    ax_.plot(a, b, 'r.', markersize=8, mec='k', mew=0.3)
    ax_.plot(a[0], b[0], 'b.', markersize=8, mec='k', mew=0.3)
    ax_.axis('off')

    #plt.scatter(x, y, c='k', s=2, edgecolor='white')
    plt.show()


def plot_predicted_positions(image, predicted_positions, l, bone):
    plt.imshow(image, cmap=plt.cm.gray)
    for s in range(cfg.num_test_patches):
        shape = np.reshape(predicted_positions[:, s], (l, 2))
        x = shape[:, 0]
        y = shape[:, 1]
        plt.plot(x, y, 'y.', markersize=1.5, alpha=0.5)

    # plt.title('Voting')

    plt.axis('off')
    output_path = os.path.join(cfg.resultsFolderPath, bone)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(output_path + '/voting_0_12.' + cfg.img_format, format=cfg.img_format, bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi)
    # plt.show()
    plt.close()


def plot_estimated_landmarks(image, landmarks):
    x, y = landmarks.T
    plt.imshow(image, cmap=plt.cm.gray)
    plt.plot(x, y, 'r.', markersize=2)
    plt.title('Estimated Landmarks')
    plt.axis('off')
    plt.show()


def kernel_density_estimation(predicted_positions):
    l = predicted_positions.shape[0] // 2
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(predicted_positions.T)
    sc = kde.score_samples(predicted_positions.T)

    max = np.argmax(np.exp(sc))
    landmarks = predicted_positions[:, max]
    landmarks = np.reshape(landmarks, (l, 2))

    return landmarks


def build_matrices(bone, scale, n_subs=0):
    path = '/' + bone + '/scale_' + scale + '/subshape_' + str(n_subs)
    t0 = time()
    d, f, c = storage.get_data(path)
    print('TIME GETTING MATRICES', time() - t0)
    D = tuple(d)
    F = tuple(f)
    C = tuple(c)

    D = np.concatenate(D, axis=1)
    F = np.concatenate(F, axis=1)
    C = np.concatenate(C, axis=1)

    return D, F, C


def compute_D_matrix(f_hat, d_tilde, c_bar, l):
    k_tilde = d_tilde.shape[1]
    k = cfg.num_test_patches
    p = np.concatenate((np.identity(k_tilde), np.zeros((k, k_tilde))))
    q = np.concatenate((np.zeros((k_tilde, k)), np.identity(k)))

    u = compute_U_matrix()

    G = compute_G_matrix(d_tilde, p, q, c_bar, u, l)
    A = compute_A_matrix(f_hat, p, q, u, l)

    d_hat = -G.dot(np.linalg.inv(A))

    d = d_hat.dot(q)

    return d


def compute_G_matrix(d_tilde, p, q, c_bar, u, l):
    k = cfg.num_test_patches
    k_tilde = p.shape[1]
    beta = cfg.beta
    # G = - (np.dot(d_tilde,p.T))/(l*k_tilde) - beta*np.dot(np.dot(c_bar,u.T),q.T)/(l*k)
    # G = -(d_tilde.dot(p.T))/(l*k_tilde) - (beta*c_bar.dot(u.T).dot(q.T))/(l*k)
    G = -(d_tilde @ p.T) / (l * k_tilde) - (beta * c_bar @ u.T @ q.T) / (l * k)

    return G


def compute_A_matrix(f_hat, p, q, u, l):
    k = cfg.num_test_patches
    k_tilde = p.shape[1]
    beta = cfg.beta
    alpha = cfg.alpha

    m = compute_M_matrix(f_hat)

    # A = (1/(l*k_tilde))* np.dot(p,p.T) + ((2*alpha)/l)*m + (beta/(l*k))*np.dot(np.dot(np.dot(q,u),u.T),q.T)
    A = (1 / (l * k_tilde)) * p @ p.T + ((2 * alpha) / l) * m + (beta / (l * k)) * q @ u @ u.T @ q.T

    return A


def compute_U_matrix():
    k = cfg.num_test_patches
    u = []
    e_matrix = np.identity(k)
    for i in range(k - 1):
        u.append(e_matrix[:, i] - e_matrix[:, i + 1])
    u = np.array(u).T

    return u


def compute_C_matrix(c, l):
    k = cfg.num_test_patches
    c_bar = []
    for i in range(k - 1):
        c_bar.append(c[:, i + 1] - c[:, i])

    c_bar = np.array(c_bar).T
    c_bar = np.tile(c_bar, (l, 1))

    return c_bar


def compute_M_matrix(f_hat):
    # s = compute_S_matrix(f_hat)
    s = compute_s_matrix(f_hat.T)

    m = csgraph.laplacian(s, normed=False)  # Laplaciano
    trace = np.trace(m)
    m /= trace
    return m


def compute_S_matrix(f_hat):
    cols = f_hat.shape[1]
    s_nearest = cfg.s_nn

    temp = np.zeros((cols, cols))
    s = np.zeros((cols, cols))
    # buscar una mejora!
    for i in range(cols):
        print('ROW', i)
        for j in range(cols):
            if i != j:
                temp[i, j] = LA.norm(f_hat[:, i] - f_hat[:, j])
        indexes = temp[i, :].argsort()[:s_nearest + 1]

        for ind in indexes[1:]:
            s[i, ind] = 1

    return s


def compute_s_matrix(f_hat):
    nbrs = NearestNeighbors(n_neighbors=cfg.s_nn, algorithm='ball_tree').fit(f_hat)
    # distances, indices = nbrs.kneighbors(f_hat)

    graph = nbrs.kneighbors_graph(f_hat).toarray()

    return graph


def show_landmarks_detected(landmarks):
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    plt.plot(x, y, 'y.', markersize=1.5)
    # plt.show()


if __name__ == '__main__':
    run()
