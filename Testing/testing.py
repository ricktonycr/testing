import multiprocessing
import os
import time
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from Testing import gradient_profiling, active_shape_model
from Testing import landmark_detection, metrics
from Tools import config as cfg, pyramid_gaussian, json_reader, data_storage


def run(bones, gt=False):
    image = io.imread(cfg.imagePath, as_gray=True)
    images_pyramid = pyramid_gaussian.get_pyramid(image)

    sigma_values = np.array(cfg.sigma_values)

    patch_sizes = cfg.testing_patch_sizes

    print('Testing Image ...', cfg.imagePath)
    f = os.path.basename(cfg.imagePath)
    fn, ext = os.path.splitext(f)
    cfg.resultsFolderPath = os.path.join(cfg.resultsFolderPath, fn)

    subshapes = None
    cfg.num_of_patches = cfg.num_test_patches  # Modificar el numero de parches a muestrear

    final_shapes = []
    gt_shapes = []
    bones_name = '_'.join(bones)
    # Segmentation for each bone (femur, pelvis)
    for bone in bones:
        times = []

        for img, sc_name, s, patch_size, idx_sc in zip(images_pyramid, cfg.scale_names, sigma_values, patch_sizes, range(len(cfg.scale_names))):
            print('Testing Scale ', sc_name)
            subshapes_ = []
            if sc_name == cfg.init_scale:
                init_flag = True
                start = time.time()
                landmarks = landmark_detection.get_estimated_landmarks(bone, img, sc_name, init_flag=init_flag)
                end = time.time()

                times.append(end - start)

                # added
                plot_segmented_shape(img, landmarks, sc_name, bone)

                subshapes = [landmarks[x:x + cfg.subs_len] for x in range(0, len(landmarks), cfg.subs_len)]
                # avoid subshapes of 1 landmark
                if len(subshapes[-1]) == 1:
                    lmrk = subshapes.pop()
                    subshapes[-1] = np.vstack((subshapes[-1], lmrk))
                subshapes = np.array(subshapes) * cfg.downScaleFactor  # upsample landmarks
                continue

            # Sequential Process
            """
            inicio = time.time()
            for n_sub, subs in enumerate(subshapes):
                print('Detecting Landmarks in Subshape', str(n_sub))

                landmarks = landmark_detection.get_estimated_landmarks(bone, img, sc_name, subs, n_sub)
                subshapes_.append(landmarks)
            fin = time.time()
            print('TIEMPO', fin - inicio)

            """

            # Parallel Process
            l = len(subshapes)
            print('LEN SUBSHAPES', l)
            num_cores = multiprocessing.cpu_count()
            # n = len(os.sched_getaffinity(0))

            pool = multiprocessing.Pool(processes=num_cores)
            try:
                start = time.time()

                subshapes_ = pool.starmap(landmark_detection.get_estimated_landmarks,
                                          zip(repeat(bone), repeat(img), repeat(sc_name), subshapes, range(l)))

            finally:
                pool.close()
                pool.join()
                end = time.time()
                times.append(end - start)

            subshapes = np.array(subshapes_)
            shape = np.vstack(subshapes)  # Updated shape
            # added
            plot_segmented_shape(img, shape, sc_name, bone)

            # ASM
            print('Computing Active Shape Model...')
            start = time.time()
            shape = active_shape_model.run(bone, shape, sc_name)
            end = time.time()
            times.append(end - start)

            plot_segmented_shape(img, shape, sc_name, bone, refined=True)

            # GRADIENT PROFILING
            if sc_name != cfg.scale_names[-1]:
                print('Computing Gradient Profiling...')
                start = time.time()
                shape = gradient_profiling.run(bone, img, s, patch_size, shape, sc_name, idx_sc)
                end = time.time()
                times.append(end - start)

            if sc_name == cfg.scale_names[-1]:
                final_shapes.append(shape)
                if gt:
                    # leer el json y comparar
                    f = os.path.basename(cfg.imagePath)
                    fn, ext = os.path.splitext(f)
                    json = os.path.join(cfg.path_json, fn + '.json')
                    shapes_gt = json_reader.get_all_landmarks_(json)
                    shape_gt = shapes_gt[cfg.bone_structures.index(bone)]
                    # shape_gt /= 2
                    gt_shapes.append(shape_gt)

                    x, y = shape_gt.T
                    plt.plot(x, y, 'g|-', ms=1.8, lw=0.5, label='Gold Standard')

                    plot_segmented_shape(img, shape, sc_name, bone, gt=True)

                    # Metrics
                    metrics.compute_segmentation_metrics(shape_gt, shape, image, bone)

            subshapes = [shape[x:x + cfg.subs_len] for x in range(0, len(shape), cfg.subs_len)]
            # avoid subshapes of 1 landmark
            if len(subshapes[-1]) == 1:
                lmrk = subshapes.pop()
                subshapes[-1] = np.vstack((subshapes[-1], lmrk))
            subshapes = np.array(subshapes)

            subshapes *= cfg.downScaleFactor

        # Save the computation time
        header = 'Computation Time for {}\n'.format(bone)
        header += 'Landmark Detection Initial Scale, Landmark Detection, ASM, Gradient Profiling for other scales (25%,50%,100%)'
        metrics.save_computation_time(times, header, bone)

    # Final segmentation
    plt.imshow(image, cmap=plt.cm.gray)
    final_shapes = np.array(final_shapes)

    for sh in final_shapes:
        x, y = sh.T
        plt.plot(x, y, 'o-', ms=2, mec='k', mew=0.3)

    plt.axis('off')
    output_path = cfg.resultsFolderPath
    plt.savefig(output_path + '/final_segmentation_' + bones_name + '_.' + cfg.img_format, format=cfg.img_format,
                bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi)
    plt.close()

    # JSW
    if len(final_shapes) % 2 == 0:  # this must be modified
        seg_distances = calculate_jsw(image, final_shapes, bones_name)
        if gt:
            gt_shapes = np.array(gt_shapes)
            gt_distances = calculate_jsw(image, gt_shapes, bones_name, gt=True)

            # Error Rates
            metrics.compute_jsw_error_rates(gt_distances, seg_distances, bones_name)


def calculate_jsw(image, shapes, bones_name, gt=False):
    landmarks_jsw = []
    for sh in shapes:
        if len(sh) == cfg.num_l_femur:
            f = sh[28:40]
            landmarks_jsw.append(f)
        elif len(sh) == cfg.num_l_pelvis:
            p = sh[10:32]
            landmarks_jsw.append(p)

    plt.imshow(image, cmap=plt.cm.gray)
    for shape in shapes:
        x, y = shape.T
        if not gt:
            plt.plot(x, y, 'r|-', ms=1.2, lw=0.5, label='Segmentation Result')

        else:
            plt.plot(x, y, 'g|-', ms=1.2, lw=0.5, label='Gold Standard')

    femurs = landmarks_jsw[::2]
    pelvises = landmarks_jsw[1::2]

    jsw_distances = []
    pair_points = []
    for femur, pelvis in zip(femurs, pelvises):
        min_distances = []
        for lmrk_p in femur:
            l_p = lmrk_p
            distances = np.linalg.norm(lmrk_p - pelvis, axis=1)

            idx_min = np.argmin(distances)
            l_f = pelvis[idx_min]

            pair_points.append([l_p, l_f])
            min_distances.append(distances[idx_min])

        ref_shape = np.array(cfg.ref_shape)
        scale_factor = np.mean(ref_shape/image.shape)
        min_distances = np.array(min_distances)*cfg.pixel_spacing*scale_factor


        for pair, i in zip(pair_points, range(len(pair_points))):
            plt.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], 'o-', ms=2.5, mec='k', mew=0.3, lw=2,
                     label='Distance ' + str(i + 1) + ': ' + str(round(min_distances[i], 1)) + 'mm')

        #min_distances = np.array(min_distances)
        jsw_distances.append(min_distances)
        print('MEAN DISTANCE', np.mean(min_distances))  # can be median,min,max...

        # delete the repeated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        # print('labels', labels)
        i = 1
        while i < len(labels):
            if labels[i] in labels[:i]:
                del (labels[i])
                del (handles[i])
            else:
                i += 1

        plt.legend(handles, labels, loc=0, fontsize='x-small')

        plt.axis('off')
        output_path = cfg.resultsFolderPath
        if not gt:
            plt.savefig(output_path + '/JSW_seg_' + bones_name + '.' + cfg.img_format, format=cfg.img_format,
                        bbox_inches='tight',
                        pad_inches=0, dpi=cfg.dpi)
        else:
            plt.savefig(output_path + '/JSW_gs_' + bones_name + '.' + cfg.img_format, format=cfg.img_format,
                        bbox_inches='tight',
                        pad_inches=0, dpi=cfg.dpi)
        plt.close()

    return jsw_distances


def plot_segmented_shape(image, landmarks, scale, bone, refined=False, gt=False):
    x, y = landmarks.T
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    output_path = os.path.join(cfg.resultsFolderPath, bone)

    if gt:
        plt.plot(x, y, 'r.-', ms=2, lw=0.5, label='Segmentation Result')
        plt.legend(loc=1, fontsize='small')
        plt.savefig(output_path + '/comparison_gs_sr_' + scale + '.' + cfg.img_format, format=cfg.img_format,
                    bbox_inches='tight',
                    pad_inches=0, dpi=cfg.dpi)

    else:
        plt.plot(x, y, 'r.-', ms=2, lw=0.5)

        if refined:
            plt.savefig(output_path + '/refined_segmented_shape_' + scale + '.' + cfg.img_format, format=cfg.img_format,
                        bbox_inches='tight',
                        pad_inches=0, dpi=cfg.dpi)
        else:
            plt.savefig(output_path + '/segmented_shape_' + scale + '.' + cfg.img_format, format=cfg.img_format,
                        bbox_inches='tight',
                        pad_inches=0, dpi=cfg.dpi)
    plt.close()


def test_folder(bones, gt):
    testing_images = os.listdir(cfg.testing_folder)
    print('Testing {} images'.format(len(list(testing_images))))

    for img in testing_images:
        cfg.imagePath = os.path.join(cfg.testing_folder, img)

        test_image(bones, gt=gt)

        print('Image {} Finished'.format(img))
        cfg.resultsFolderPath, _ = os.path.split(cfg.resultsFolderPath)


def test_image(bones_, gt):
    start = time.time()
    run(bones_, gt=gt)
    end = time.time()
    metrics.save_computation_time([end - start], 'Total Time Elapsed')


if __name__ == '__main__':
    bones_side = cfg.rigth_side
    ground_truth = True

    test_folder(bones_side, ground_truth)

    print('TESTING FINISHED')

