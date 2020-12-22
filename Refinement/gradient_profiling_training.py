from skimage import io
from Refinement import gradient_profiling as gp
import numpy as np
import os
from Tools import config as cfg, pyramid_gaussian
from Tools import data_storage


def run():

    images_list = os.listdir(cfg.datasetRoot)
    images_list = filter(lambda element: '.JPG' in element, images_list)

    dataset_images = os.listdir(cfg.datasetRoot)
    dataset_images = filter(lambda element: '.JPG' in element, dataset_images)
    count = len(list(dataset_images))

    print('Training', count, 'X-Ray Images')

    femur_shape = np.array([(141.33333333333334, 610.6666666666666), (148.0, 604.0), (155.33333333333334, 598.6666666666666), (163.33333333333334, 592.0), (169.33333333333334, 584.0), (172.66666666666666, 575.3333333333334), (174.66666666666666, 566.6666666666666), (173.5, 556.5), (168.66666666666666, 548.0), (165.33333333333334, 540.0), (168.0, 533.0), (172.0, 527.5), (176.0, 521.5), (181.5, 515.5), (186.5, 509.5), (190.5, 504.0), (196.5, 499.5), (202.5, 495.0), (209.0, 492.0), (219.0, 493.5), (231.0, 495.0), (244.0, 495.5), (254.0, 489.0), (262.5, 480.0), (268.5, 471.5), (274.0, 462.5), (275.5, 451.0), (275.5, 438.0), (274.0, 418.5), (273.0, 406.5), (267.5, 396.5), (259.5, 387.5), (249.0, 381.0), (238.0, 376.5), (226.0, 373.5), (214.0, 372.5), (200.5, 374.5), (187.5, 378.5), (176.5, 384.0), (166.0, 392.5), (160.0, 402.5), (155.5, 414.0), (149.5, 422.5), (138.5, 425.0), (126.0, 425.0), (115.5, 424.5), (105.5, 420.0), (99.5, 414.0), (91.5, 405.5), (81.0, 400.0), (70.5, 399.5), (61.5, 410.0), (51.0, 420.5), (43.5, 431.0), (38.5, 443.5), (31.0, 456.0), (26.0, 468.0), (22.5, 483.0), (25.5, 496.5), (32.5, 507.0), (37.5, 517.5), (41.5, 528.5), (44.5, 540.5), (46.5, 552.0), (48.0, 563.0), (49.0, 574.5), (50.0, 585.0), (51.0, 598.0), (51.0, 609.0)])
    db = cfg.database_root_gp
    data_storage.create_database(db)

    for filename in images_list:
        print('Processing Image: %s...' %filename)
        image_path = cfg.datasetRoot+'/'+filename
        image = io.imread(image_path, as_gray=True)
        fn = filename[0:-4]
        g = data_storage.create_group(fn)
        # Falta leer un JSON para cada estructura y cada forma
        shape = femur_shape # shape simulada

        pyramid = pyramid_gaussian.get_pyramid(image)
        sigma = cfg.sigma_values
        patch_size = cfg.patch_sizes
        scales = cfg.scale_names
        matrix = []
        for p,s,ps,sc in zip(pyramid, sigma, patch_size, scales):
            sg = data_storage.create_group(db, 'scale_' + sc, g)
            img_grad = gp.get_gaussian_gradient_magnitude(p,s)
            # Modificar mtx
            patches = gp.sample_patches(img_grad, shape, (ps, ps))

            for patch,i in zip(patches, range(1, len(patches)+1)):
                data_storage.save_data(sg, 'patch_'+str(i), patch)

            shape /= cfg.downScaleFactor # escalamos las posiciones de los landmarks


if __name__ == '__main__':
    run()