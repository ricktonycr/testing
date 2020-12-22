import os
import time

import numpy as np
from skimage import io

import extract_features
from Refinement import gradient_profiling as grad_prof
from Refinement import principal_component_analysis as pca
from Tools import config as cfg, json_reader as reader, find_displacement, pyramid_gaussian, sample_patches
from Tools import data_storage as storage


def run():
    gp_DB = cfg.database_root_gp
    storage.create_database()  # database for landmark detection
    storage.create_database(gp_DB)  # DB for gradient profiling

    # images = glob.glob(cfg.datasetRoot+'/*.png')

    # List of X-Ray images in PNG format
    images_list = os.listdir(cfg.datasetRoot)
    images_list = filter(lambda element: '.JPG' in element, images_list)
    """
    images = []
    for img in images_list:
        if img.endswith('.png') or img.endswith('.JPG'):
            images.append(img)
    
    """
    ###
    # images_list = [img for img in os.listdir(cfg.datasetRoot) if img.endswith('.JPG')]

    ##############################################################
    dataset_images = os.listdir(cfg.datasetRoot)
    dataset_images = filter(lambda element: '.JPG' in element, dataset_images)
    count = len(list(dataset_images))

    print('Training', count, 'X-Ray Images')

    # List for each bone structure
    R_femurs = []
    L_femurs = []
    R_pelvis = []
    L_pelvis = []

    for filename in images_list:
        print('Training Image: {0}...'.format(filename))

        image_path = os.path.join(cfg.datasetRoot, filename)
        image = io.imread(image_path, as_gray=True)
        fn = filename[0:-4]

        pyramid = pyramid_gaussian.get_pyramid(image)

        json_file = os.path.join(cfg.path_json, fn + '.gsmdc')
        structures = reader.get_all_subshapes(json_file)

        im = storage.create_group(name=fn)
        gp_gr = storage.create_group(gp_DB, fn)

        for subshapes_, bone_name in zip(structures, cfg.bone_structures):
            # For ASM
            print('STRUCTURES', len(structures))
            shape_ = np.array(np.vstack(subshapes_))
            print('size', len(shape_))
            if bone_name == cfg.bone_structures[0]:
                R_femurs.append(shape_)
            elif bone_name == cfg.bone_structures[1]:
                R_pelvis.append(shape_)
            elif bone_name == cfg.bone_structures[2]:
                L_femurs.append(shape_)
            elif bone_name == cfg.bone_structures[3]:
                L_pelvis.append(shape_)

            # For initial scale
            subshapes = subshapes_ / (cfg.downScaleFactor ** (len(cfg.scale_names) - 1))
            shape = shape_ / (cfg.downScaleFactor ** (len(cfg.scale_names) - 1))
            init_flag = False

            g = storage.create_group(name=bone_name, parent=im)
            gp_group = storage.create_group(gp_DB, name=bone_name, parent=gp_gr)

            for img, scale, sigma, patch_size in zip(pyramid, cfg.scale_names, cfg.sigma_values, cfg.patch_sizes):
                # LANDMARK DETECTION DATA
                print('Scale: {0} - Subshapes: {1} '.format(scale, len(subshapes)))
                sg = storage.create_group(name='scale_' + scale, parent=g)
                if scale == cfg.init_scale:
                    init_flag = True
                    subshapes = np.array([shape])

                for i in range(len(subshapes)):
                    subs_g = storage.create_group(name='subshape_' + str(i), parent=sg)

                    try:
                        D, F, C = training_an_image(img, subshapes[i], init_flag)
                        storage.save_data(subs_g, 'D_' + fn, D)
                        storage.save_data(subs_g, 'F_' + fn, F)
                        storage.save_data(subs_g, 'C_' + fn, C)

                        if init_flag:
                            subshapes = subshapes_ / (cfg.downScaleFactor ** (len(cfg.scale_names) - 1))
                            init_flag = False
                    except:
                        print('Except subshape', i)

                # GRADIENT PROFILING DATA
                if scale != cfg.init_scale:
                    gp_g = storage.create_group(gp_DB, 'scale_' + scale, gp_group)
                    img_grad = grad_prof.get_gaussian_gradient_magnitude(img, sigma)
                    # img_grad = img

                    patches = grad_prof.sample_patches(img_grad, shape, (patch_size, patch_size))

                    mtx_patches = [np.ravel(patch) for patch in patches]

                    mtx_patches = np.array(mtx_patches).transpose()

                    storage.save_data(gp_g, 'patches', mtx_patches, gp_DB)

                shape *= cfg.downScaleFactor
                subshapes *= cfg.downScaleFactor

    bone_structures = [R_femurs, R_pelvis, L_femurs, L_pelvis]

    # ACTIVE SHAPE MODEL DATA
    for bone_shapes, model in zip(bone_structures, cfg.models):
        shapes = np.array(bone_shapes)
        if len(shapes) > 1:
            pca.run(shapes, model)

    print('Training Finished')


def training_an_image(image, landmarks, init_flag):
    # Sample patches and patch centres
    patches, patch_centres_matrix = sample_patches.create_patches_randomly(image, landmarks, init_flag)

    # Find displacements
    displacements_matrix = find_displacement.calculate_displacement_matrix(landmarks, patch_centres_matrix.T)

    # Extract feature vectors
    features_matrix = extract_features.extractFeaturesForPatches(patches)  # mxk

    return displacements_matrix, features_matrix, patch_centres_matrix


if __name__ == '__main__':

    ini = time.process_time()
    run()
    fin = time.process_time()
    print('TOTAL TIME: ', fin - ini)
