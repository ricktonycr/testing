import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import path
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.spatial.distance import dice, euclidean, directed_hausdorff, jaccard, pdist
from skimage import io, draw

from Tools import config as cfg


def euclidean_distance(shape1, shape2):
    if shape1.shape != shape2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    distance = np.linalg.norm(shape1 - shape2)

    return distance


def average_euclidean_distance(shape1, shape2):
    if shape1.shape != shape2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    distances = np.linalg.norm(shape1 - shape2, axis=1)

    return np.mean(distances)


def hausdorff_distance(shape1, shape2):
    hausdorff_dist = max(directed_hausdorff(shape1, shape2)[0], directed_hausdorff(shape2, shape1)[0])
    return hausdorff_dist


def dice_coefficient(shape1, shape2):
    img = io.imread(cfg.imagePath, as_gray=True)
    img_shape = img.shape[::-1]
    gt_x, gt_y = shape1.T
    gt = poly2mask(gt_x, gt_y, img_shape)

    seg_x, seg_y = shape2.T
    seg = poly2mask(seg_x, seg_y, img_shape)

    intersection = np.logical_and(gt, seg)
    dice_coeff = 2. * np.sum(intersection) / (np.sum(gt) + np.sum(seg))

    return dice_coeff


def iou(shape1, shape2):
    img = io.imread(cfg.imagePath, as_gray=True)
    ### Getting the ROI ###
    closed_path = path.Path(shape1)
    closed_path_2 = path.Path(shape2)

    # Get the points that lie within the closed path
    idx = np.array([[(i, j) for i in range(img.shape[1])] for j in range(img.shape[0])]).reshape(np.prod(img.shape), 2)
    gt = closed_path.contains_points(idx).reshape(img.shape)
    seg = closed_path_2.contains_points(idx).reshape(img.shape)

    # Invert the mask and apply to the image
    # mask = np.invert(mask)
    # mask2 = np.invert(mask2)
    # masked_data = ma.array(img.copy(), mask=mask) # ROI
    # masked_data2 = ma.array(img.copy(), mask=mask2) # ROI

    # plt.imshow(masked_data)
    # plt.title('masked_data')
    # plt.show()

    # plt.imshow(masked_data2)
    # plt.title('masked_data2')
    # plt.show()

    intersection = np.logical_and(gt, seg)
    union = np.logical_or(gt, seg)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def iou_metric(shape1, shape2, img, bone, plot=False):
    img_shape = img.shape[::-1]
    gt_x, gt_y = shape1.T
    gt = poly2mask(gt_x, gt_y, img_shape)

    seg_x, seg_y = shape2.T
    seg = poly2mask(seg_x, seg_y, img_shape)

    intersection = np.logical_and(gt, seg)
    union = np.logical_or(gt, seg)
    iou_score = np.sum(intersection) / np.sum(union)

    if plot:
        plot_masks(shape1, shape2, img, bone)

    return iou_score


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def plot_shapes(shape1, shape2):
    s1x, s1y = shape1.T
    s2x, s2y = shape2.T

    img = io.imread(cfg.imagePath, as_gray=True)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.plot(s1x, s1y, '.g', label='shape 1')
    plt.plot(s2x, s2y, '.r', label='shape 2')
    plt.legend()
    plt.show()


def plot_masks(shape1, shape2, image, bone):
    s1 = np.append(shape1, [shape1[0]], axis=0)
    s2 = np.append(shape2, [shape2[0]], axis=0)

    # drawing the contours
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    x, y = s1.T
    plt.plot(x, y, '-g', lw=1, label='Gold Standard')

    x2, y2 = s2.T
    plt.plot(x2, y2, '-r', lw=1, label='Segmentation Result')
    output_path = os.path.join(cfg.resultsFolderPath, bone)
    plt.legend(loc=0, fontsize='small')
    plt.savefig(output_path + '/final_contours.' + cfg.img_format, format=cfg.img_format, bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi)
    # plt.show()
    plt.close()

    # Drawing the areas of the masks
    fig, ax = plt.subplots()

    patches = []

    poly1 = Polygon(shape1, closed=True, color='green', label='Gold Standard')
    poly2 = Polygon(shape2, closed=True, color='red', label='Segmentation Result')
    patches.append(poly1)
    patches.append(poly2)

    p = PatchCollection(patches, alpha=0.60, match_original=True)
    ax.add_collection(p)
    ax.legend(handles=patches, loc=0, fontsize='small')
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(output_path + '/final_masks.' + cfg.img_format, format=cfg.img_format, bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi)
    # plt.show()
    plt.close()


def plot_jsw_error_rates():
    result = list(Path(cfg.resultsFolderPath).rglob("jsw_error_rates.[tT][xX][tT]"))

    jsw_distances = []

    for f in result:
        data = np.loadtxt(f)
        jsw_distances.append(data)

    jsw_distances = np.vstack(jsw_distances)
    jsw = jsw_distances[:, :12]
    other_metrics = jsw_distances[:, 12:]

    medianprops = dict(linewidth=1.5, color='red')
    boxprops = dict(linewidth=1.5, color='#0277bd')
    whiskerprops = dict(linewidth=1.2, linestyle='--', color='#0277bd')
    capprops = dict(linewidth=1.4)

    plt.xlabel('Distances')
    plt.ylabel('Absolute Error (mm)')
    plt.boxplot(jsw, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops)
    plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
    plt.savefig(cfg.resultsFolderPath + '/JSW_error_rate_distances.' + cfg.img_format, format=cfg.img_format,
                bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi + 100)
    plt.show()

    labels = ['Mean', 'Median', 'Minimum', 'Maximum']

    plt.xlabel('Metric')
    plt.ylabel('Absolute Error (mm)')
    plt.boxplot(other_metrics, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops)
    plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.savefig(cfg.resultsFolderPath + '/JSW_error_rate_other_metrics.' + cfg.img_format, format=cfg.img_format,
                bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi + 100)
    plt.show()

    #### Signed errors
    result = list(Path(cfg.resultsFolderPath).rglob("jsw_signed_error_rates.[tT][xX][tT]"))

    jsw_distances = []

    for f in result:
        data = np.loadtxt(f)
        jsw_distances.append(data)

    jsw_distances = np.vstack(jsw_distances)
    jsw = jsw_distances[:, :12]
    other_metrics = jsw_distances[:, 12:]

    plt.xlabel('Landmarks')
    plt.ylabel('Signed Error (mm)')
    plt.boxplot(jsw, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops)
    plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
    plt.savefig(cfg.resultsFolderPath + '/JSW_signed_error_rate_distances.' + cfg.img_format, format=cfg.img_format,
                bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi + 100)
    plt.show()

    labels = ['Mean', 'Median', 'Minimum', 'Maximum']

    plt.xlabel('Metric')
    plt.ylabel('Signed Error (mm)')
    plt.boxplot(other_metrics, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops)
    plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.savefig(cfg.resultsFolderPath + '/JSW_signed_error_rate_other_metrics.' + cfg.img_format, format=cfg.img_format,
                bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi + 100)
    plt.show()


def plot_segmentation_metrics():
    femur_files = list(Path(cfg.resultsFolderPath).rglob("*femur/metrics.[tT][xX][tT]"))
    pelvis_files = list(Path(cfg.resultsFolderPath).rglob("*pelvis/metrics.[tT][xX][tT]"))

    files = [femur_files, pelvis_files]
    for i, fs in enumerate(files):
        if i == 0:
            name = 'femur'
        else:
            name = 'pelvis'

        all_metrics = []
        for f in fs:
            metrics = np.loadtxt(f)
            all_metrics.append(metrics)

        all_metrics = np.vstack(all_metrics)
        iou_dice = all_metrics[:, :2]
        dice = all_metrics[:, 1]
        hausdorff = all_metrics[:, 2]
        eucl = all_metrics[:, 3]
        avg_eucl = all_metrics[:, 4]

        medianprops = dict(linewidth=1.5, color='red')
        boxprops = dict(linewidth=1.5, color='#0277bd')
        whiskerprops = dict(linewidth=1.2, linestyle='--', color='#0277bd')
        capprops = dict(linewidth=1.4)

        labels = ['Jaccard Index', 'DICE Coefficient']

        plt.xlabel('Metric')
        plt.ylabel('Similarity Measure')
        plt.boxplot(iou_dice, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops)
        plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.savefig(cfg.resultsFolderPath + '/Jaccard_Dice_metrics_' + name + '.' + cfg.img_format,
                    format=cfg.img_format, bbox_inches='tight',
                    pad_inches=0, dpi=cfg.dpi + 100)
        plt.show()

        plt.xlabel('Metric')
        plt.ylabel('Similarity Measure')
        plt.boxplot(dice, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops)
        plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
        plt.xticks([1], ['DICE Coefficient'])
        plt.savefig(cfg.resultsFolderPath + '/Dice_metric_' + name + '.' + cfg.img_format, format=cfg.img_format,
                    bbox_inches='tight', pad_inches=0, dpi=cfg.dpi + 100)
        plt.show()

        labels = ['Avg Euclidean Distance', 'Hausdorff Distance']

        plt.xlabel('Metric')
        plt.ylabel('Distance')
        plt.boxplot([avg_eucl, hausdorff], showfliers=False, medianprops=medianprops, boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops)
        plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.savefig(cfg.resultsFolderPath + '/Avg_Euclidean_Hausdorff_metrics_' + name + '.' + cfg.img_format,
                    format=cfg.img_format, bbox_inches='tight',
                    pad_inches=0, dpi=cfg.dpi + 100)
        plt.show()

        plt.xlabel('Metric')
        plt.ylabel('Distance')
        plt.boxplot(avg_eucl, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops)
        plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
        plt.xticks([1], ['Average Euclidean Distance'])
        plt.savefig(cfg.resultsFolderPath + '/Avg_Euclidean_metric_' + name + '.' + cfg.img_format,
                    format=cfg.img_format,
                    bbox_inches='tight', pad_inches=0, dpi=cfg.dpi + 100)
        plt.show()

        plt.xlabel('Metric')
        plt.ylabel('Distance')
        plt.boxplot(hausdorff, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops)
        plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
        plt.xticks([1], ['Hausdorff Distance'])
        plt.savefig(cfg.resultsFolderPath + '/Hausdorff_metric_' + name + '.' + cfg.img_format, format=cfg.img_format,
                    bbox_inches='tight', pad_inches=0, dpi=cfg.dpi + 100)
        plt.show()

        plt.xlabel('Metric')
        plt.ylabel('Distance')
        plt.boxplot(eucl, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                    capprops=capprops)
        plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')
        plt.xticks([1], ['Euclidean Distance'])
        plt.savefig(cfg.resultsFolderPath + '/Euclidean_metric_' + name + '.' + cfg.img_format, format=cfg.img_format,
                    bbox_inches='tight', pad_inches=0, dpi=cfg.dpi + 100)
        plt.show()


def get_computation_times():
    total_time = list(Path(cfg.resultsFolderPath).rglob("H*/computation_time.[tT][xX][tT]"))
    femur_files = list(Path(cfg.resultsFolderPath).rglob("*femur/computation_time.[tT][xX][tT]"))
    pelvis_files = list(Path(cfg.resultsFolderPath).rglob("*pelvis/computation_time.[tT][xX][tT]"))

    total_times = []
    for t in total_time:
        time_ = np.loadtxt(t)
        total_times.append(time_)

    total_times = np.hstack(total_times)
    print('Average Computation Time: ', total_times.mean())

    femur_times = []
    pelvis_times = []
    for f, p in zip(femur_files, pelvis_files):
        time_f = np.loadtxt(f)
        time_p = np.loadtxt(p)

        femur_times.append(time_f)
        pelvis_times.append(time_p)

    femur_times = np.vstack(femur_times)
    pelvis_times = np.vstack(pelvis_times)

    times_f = np.mean(femur_times, axis=0)
    times_p = np.mean(pelvis_times, axis=0)
    print('Average Computation Time for Femur', times_f)
    print('Average Computation Time for Pelvis', times_p)

    pos = np.arange(len(times_f))
    bar_width = 0.25
    labels = ['Scale 0.12', 'Scale 0.25', 'Scale 0.5', 'Scale 1']
    landmark_detection_times = np.array([(times_f[i], times_p[i]) for i in [0, 1, 4, 7]])
    asm_times = np.array([[times_f[i], times_p[i]] for i in [2, 5, 8]])
    gradient_prof_times = np.array([[times_f[i], times_p[i]] for i in [3, 6]])

    ld_range = np.array([0, .5, 1.5, 2.5])
    ticks_pos = np.array([0, 0.75, 1.75, 2.625])
    plt.bar(x=ld_range, height=landmark_detection_times[:, 0], width=bar_width, label='Landmark Detection', zorder=2)
    plt.bar(x=ld_range[1:] + bar_width, height=asm_times[:, 0], width=bar_width, label='Active Shape Model', zorder=2)
    plt.bar(x=ld_range[1:3] + bar_width * 2, height=gradient_prof_times[:, 0], width=bar_width,
            label='Gradient Profiling', zorder=2)

    plt.xticks(ticks_pos, labels)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel('Scales')
    plt.ylabel('Computation Time (s)')
    plt.legend(loc=0, fontsize='small')
    plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')

    plt.savefig(cfg.resultsFolderPath + '/Computation_time_femur.' + cfg.img_format, format=cfg.img_format,
                bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi + 100)
    plt.show()

    plt.bar(x=ld_range, height=landmark_detection_times[:, 1], width=bar_width, label='Landmark Detection', zorder=2)
    plt.bar(x=ld_range[1:] + bar_width, height=asm_times[:, 1], width=bar_width, label='Active Shape Model', zorder=2)
    plt.bar(x=ld_range[1:3] + bar_width * 2, height=gradient_prof_times[:, 1], width=bar_width,
            label='Gradient Profiling', zorder=2)

    plt.xticks(ticks_pos, labels)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel('Scales')
    plt.ylabel('Computation Time (s)')
    plt.legend(loc=0, fontsize='small')
    plt.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, axis='y')

    plt.savefig(cfg.resultsFolderPath + '/Computation_time_pelvis.' + cfg.img_format, format=cfg.img_format,
                bbox_inches='tight',
                pad_inches=0, dpi=cfg.dpi + 100)
    plt.show()


def save_computation_time(time_elapsed, header, bone='bone'):
    name = 'computation_time.txt'
    if bone == 'bone':
        filename = os.path.join(cfg.resultsFolderPath, name)
    else:
        filename = os.path.join(cfg.resultsFolderPath, bone, name)

    if not os.path.exists(filename):
        file = open(filename, 'w')
    else:
        file = open(filename, 'a')

    time_elapsed = np.column_stack(time_elapsed)
    np.savetxt(file, time_elapsed, header=header)
    file.close()


def save_seg_metrics(iou, dice, euclidean, avg_euclidean, hausdorff, bone):
    name = 'metrics.txt'
    path = cfg.resultsFolderPath
    filename = os.path.join(path, bone, name)
    if not os.path.exists(filename):
        file = open(filename, 'w')
    else:
        file = open(filename, 'a')

    header = 'METRICS: \n IoU, DICE Coefficient, Hausdorff Distance, Euclidean Distance, Average Euclidean Distance'
    values = np.column_stack((iou, dice, hausdorff, euclidean, avg_euclidean))

    np.savetxt(file, values, header=header)
    file.close()


def save_jsw_error_rate(error_rates, header, signed_flag=False):
    if signed_flag:
        name = 'jsw_signed_error_rates.txt'
    else:
        name = 'jsw_error_rates.txt'
    path = cfg.resultsFolderPath
    filename = os.path.join(path, name)
    if not os.path.exists(filename):
        file = open(filename, 'w')
    else:
        file = open(filename, 'a')

    np.savetxt(file, error_rates, header=header)
    file.close()


def compute_segmentation_metrics(shape_gt, shape_seg, image, bone):
    euclidean = euclidean_distance(shape_gt, shape_seg)
    avg_euclidean = average_euclidean_distance(shape_gt, shape_seg)
    hausdorff = hausdorff_distance(shape_gt, shape_seg)
    dice = dice_coefficient(shape_gt, shape_seg)
    iou = iou_metric(shape_gt, shape_seg, image, bone, plot=True)
    save_seg_metrics(iou, dice, euclidean, avg_euclidean, hausdorff, bone)  # Saving for each bone

    print(
        'METRICS: \n Euclidean Distance: {0} \n Average Euclidean Distance: {1} \n Hausdorff Distance: {2} \n DICE Coefficient: {3} '
        '\n IoU: {4}'.format(euclidean, avg_euclidean, hausdorff, dice, iou))


def compute_jsw_error_rates(gt_distances, seg_distances, bones_name):
    for gt_d, seg_d in zip(gt_distances, seg_distances):
        mean_gt, mean_seg = np.mean((gt_d, seg_d), axis=1)
        median_gt, median_seg = np.median((gt_d, seg_d), axis=1)
        min_gt, min_seg = np.min((gt_d, seg_d), axis=1)
        max_gt, max_seg = np.max((gt_d, seg_d), axis=1)

        jsw_error_rate = np.abs(seg_d - gt_d)
        mean_error_rate = np.abs(mean_seg - mean_gt)
        median_error_rate = np.abs(median_seg - median_gt)
        min_error_rate = np.abs(min_seg - min_gt)
        max_error_rate = np.abs(max_seg - max_gt)
        # Signed errors
        jsw_signed_error_rate = seg_d - gt_d
        mean_signed_error_rate = mean_seg - mean_gt
        median_signed_error_rate = median_seg - median_gt
        min_signed_error_rate = min_seg - min_gt
        max_signed_error_rate = max_seg - max_gt

        error_rates = 'Error Rates for {0}\n' \
                      'JSW Error Rate: {1}\n' \
                      'Mean Error Rate: {2}\n' \
                      'Median Error Rate: {3}\n' \
                      'Minimum Error Rate: {4}\n' \
                      'Maximum Error Rate: {5}\n\n'.format(bones_name, str(list(jsw_error_rate)), str(mean_error_rate),
                                                           str(median_error_rate), str(min_error_rate),
                                                           str(max_error_rate))

        print(error_rates)

        signed_error_rates = 'Signed Error Rates for {0}\n' \
                             'JSW Error Rate: {1}\n' \
                             'Mean Error Rate: {2}\n' \
                             'Median Error Rate: {3}\n' \
                             'Minimum Error Rate: {4}\n' \
                             'Maximum Error Rate: {5}\n\n'.format(bones_name, str(list(jsw_signed_error_rate)),
                                                                  str(mean_signed_error_rate),
                                                                  str(median_signed_error_rate),
                                                                  str(min_signed_error_rate),
                                                                  str(max_signed_error_rate))

        print(signed_error_rates)

        header = 'Error Rates for {0}\n'.format(bones_name)
        header += 'JSW Error Rate (12 vals), Mean Error Rate, Median Error Rate, Minimum Error Rate, Maximum Error Rate'
        values = np.append(jsw_error_rate, [mean_error_rate, median_error_rate, min_error_rate, max_error_rate])
        values = np.column_stack(values)
        save_jsw_error_rate(values, header)

        header = 'Signed Error Rates for {0}\n'.format(bones_name)
        header += 'JSW Error Rate (12 vals), Mean Error Rate, Median Error Rate, Minimum Error Rate, Maximum Error Rate'
        values = np.append(jsw_signed_error_rate,
                           [mean_signed_error_rate, median_signed_error_rate, min_signed_error_rate,
                            max_signed_error_rate])
        values = np.column_stack(values)
        save_jsw_error_rate(values, header, signed_flag=True)


if __name__ == '__main__':
    x1 = np.array([(141.33333333333334, 610.6666666666666), (148.0, 604.0), (155.33333333333334, 598.6666666666666),
                   (163.33333333333334, 592.0), (169.33333333333334, 584.0), (172.66666666666666, 575.3333333333334),
                   (174.66666666666666, 566.6666666666666), (173.5, 556.5), (168.66666666666666, 548.0),
                   (165.33333333333334, 540.0), (168.0, 533.0), (172.0, 527.5), (176.0, 521.5), (181.5, 515.5),
                   (186.5, 509.5), (190.5, 504.0), (196.5, 499.5), (202.5, 495.0), (209.0, 492.0), (219.0, 493.5),
                   (231.0, 495.0), (244.0, 495.5), (254.0, 489.0), (262.5, 480.0), (268.5, 471.5), (274.0, 462.5),
                   (275.5, 451.0), (275.5, 438.0), (274.0, 418.5), (273.0, 406.5), (267.5, 396.5), (259.5, 387.5),
                   (249.0, 381.0), (238.0, 376.5), (226.0, 373.5), (214.0, 372.5), (200.5, 374.5), (187.5, 378.5),
                   (176.5, 384.0), (166.0, 392.5), (160.0, 402.5), (155.5, 414.0), (149.5, 422.5), (138.5, 425.0),
                   (126.0, 425.0), (115.5, 424.5), (105.5, 420.0), (99.5, 414.0), (91.5, 405.5), (81.0, 400.0),
                   (70.5, 399.5), (61.5, 410.0), (51.0, 420.5), (43.5, 431.0), (38.5, 443.5), (31.0, 456.0),
                   (26.0, 468.0), (22.5, 483.0), (25.5, 496.5), (32.5, 507.0), (37.5, 517.5), (41.5, 528.5),
                   (44.5, 540.5),
                   (46.5, 552.0), (48.0, 563.0), (49.0, 574.5), (50.0, 585.0), (51.0, 598.0), (51.0, 609.0)])

    x2 = np.array([(135.0, 431.0), (139.5, 425.5), (143.5, 421.0), (147.0, 416.0), (150.0, 411.5), (152.5, 405.5),
                   (153.5, 400.0), (153.5, 393.5), (150.0, 388.5), (146.0, 383.0), (148.5, 378.5), (150.5, 374.5),
                   (152.0, 370.0), (155.0, 366.0), (158.5, 362.5), (161.0, 358.0), (164.0, 354.5), (169.0, 352.0),
                   (174.5, 351.0), (183.6, 352.4), (192.0, 351.6), (199.6, 346.0), (205.2, 339.6), (209.6, 331.2),
                   (212.0, 322.4), (212.4, 312.0), (210.8, 302.8), (206.8, 293.6), (201.6, 287.6), (198.0, 280.0),
                   (192.0, 275.6), (184.8, 271.2), (175.6, 268.8), (166.0, 268.0), (157.6, 268.4), (149.6, 269.6),
                   (142.0, 273.2), (136.4, 277.6), (132.4, 282.4), (128.4, 287.6), (125.6, 293.6), (125.2, 300.8),
                   (121.2, 306.0), (115.6, 311.2), (106.4, 311.2), (97.6, 309.6), (88.4, 306.0), (82.8, 299.6),
                   (78.8, 292.0), (67.2, 290.4), (56.8, 290.8), (50.4, 296.4), (50.8, 305.6), (47.2, 315.6),
                   (44.8, 324.8),
                   (41.2, 334.8), (37.6, 344.0), (35.2, 354.8), (36.4, 364.8), (41.6, 369.6), (46.8, 375.6),
                   (49.6, 381.2),
                   (52.8, 387.6), (56.8, 394.0), (58.8, 401.2), (61.6, 408.4), (63.2, 416.4), (64.8, 423.6),
                   (64.8, 430.0)])

    x2_ = x2 * 1.015
    x3 = x1

    plot_shapes(x2, x2_)

    e = euclidean_distance(x2, x2_)
    print('euclidean 1', e)

    e2 = euclidean(x2.flatten(), x2_.flatten())
    print('euclidean 2', e2)

    d = dice_coefficient(x2, x2_)
    print('DICE ', d)

    img = io.imread(cfg.imagePath, as_gray=True)
    # print('IMAGE',img.shape[::-1])

    x, y = x2.T
    mask1 = poly2mask(x, y, img.shape[::-1])
    # print('MASK1',mask1.shape)
    # plt.imshow(mask1)
    # plt.show()
    x_, y_ = x2_.T
    mask2 = poly2mask(x_, y_, img.shape[::-1])
    # print('MASK2', mask2.shape)
    # plt.imshow(mask2)
    # plt.show()

    dsc = dice(mask1.flatten(), mask2.flatten())
    print('DICE DISSIMILARITY', dsc)

    # iou_score = iou_metric(x2, x2_, img,bone='test', plot=True)
    # iou_score = iou(x2, x2_)
    # print('IoU Score',iou_score)

    ji = jaccard(mask1.flatten(), mask2.flatten())
    print('JACCARD', (1 - ji))

    h = directed_hausdorff(x2, x2_)[0]
    h2 = directed_hausdorff(x2_, x2)[0]
    hausdorff_dist = max(h, h2)

    print('hausdorff distance', hausdorff_dist)

    X = np.vstack((x2.flatten(), x2_.flatten()))
    # X = X.T
    print(X.shape)
    mh = pdist(X, 'euclidean')
    print(mh)

    mean_1 = np.mean(x2, axis=0)
    mean_2 = np.mean(x2_, axis=0)

    dis = euclidean(mean_1, mean_2)
    print('distance', dis)

    """
    de = cdist(mask1, mask2, 'jaccard')
    print('cdist',de)
    
    ec = euclidean_distances(x2, x2_)
    print('sklearn', ec)

    d = np.sqrt(np.sum((x2-x2_)**2, axis=1))
    print(d)

    distances = (x2 - x2_) ** 2
    distances = distances.sum(axis=-1)
    distances = np.sqrt(distances)

    print('dis', distances)
    """
    plot_jsw_error_rates()
    plot_segmentation_metrics()
    get_computation_times()
