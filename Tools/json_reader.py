import json

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from Tools import config as cfg


def get_all_subshapes(filename):
    shapes = get_all_landmarks_(filename)
    subshapes_per_image = []
    for s in shapes:
        subshapes = [s[x:x + cfg.subs_len] for x in range(0, len(s), cfg.subs_len)]
        # avoid subshapes of 1 landmark
        if len(subshapes[-1]) == 1:
            lmrk = subshapes.pop()
            subshapes[-1] = np.vstack((subshapes[-1], lmrk))
        subshapes = np.array(subshapes)
        subshapes_per_image.append(subshapes)
    subshapes_per_image = np.array(subshapes_per_image)
    return subshapes_per_image


def get_subshape(filename, idx):
    data = json.load(open(filename))
    subshape = []
    for l in data['landmarkSet']:
        if l['group'] == idx:
            subshape.append((l['positionX'], l['positionY']))

    return subshape


def get_all_landmarks(filename):
    data = json.load(open(filename))
    shapes = []

    try:
        landmarks = []
        for lm in data['landmarkSet']:
            landmarks.append((lm['positionX'], lm['positionY']))
        shapes.append(landmarks)
    except:
        for shape in data['landmarks']:
            landmarks = []
            for l in shape:
                landmarks.append((l['positionX'], l['positionY']))
            landmarks = np.array(landmarks)
            shapes.append(landmarks)
    shapes = np.array(shapes)
    return shapes


def get_all_landmarks_(filename):
    data = json.load(open(filename))
    shapes = []
    femur_r = []
    pelvis_r = []
    femur_l = []
    pelvis_l = []

    xx = data['imageX']
    rel = 600.0/xx
    yy = data['imagey']

    for l in data['landmarks']:
        if l['id'] < 29:
            femur_r.append((l['x']*rel, l['y']*rel))
        elif l['id'] >= 140 and l['id'] < 162:
            femur_l.append((l['x']*rel, l['y']*rel))
        elif l['id'] >= 183 and l['id'] < 204:
            pelvis_r.append((l['x']*rel, l['y']*rel))
        elif l['id'] >= 162 and l['id'] < 183:
            pelvis_l.append((l['x']*rel, l['y']*rel))

    femur_r = np.array(femur_r)
    pelvis_r = np.array(pelvis_r)
    femur_l = np.array(femur_l)
    pelvis_l = np.array(pelvis_l)

    shapes.extend([femur_r, pelvis_r, femur_l, pelvis_l])
    shapes = np.array(shapes)
    return shapes


def get_shapes(filename):
    data = json.load(open(filename))
    shapes = []
    landmarks = []
    try:
        for lm in data['landmarkSet']:
            landmarks.append((lm['positionX'], lm['positionY']))
    except:
        for shape in data['landmarks']:
            for s in shape:
                landmarks.append((s['positionX'], s['positionY']))
        landmarks = np.array(landmarks)
        shapes.append(landmarks)
    shapes = np.array(shapes)
    return shapes


def get_shapes_(filename):
    data = json.load(open(filename))
    shapes = []

    for i in range(1,5):
        landmarks = []
        print(i)
        for lm in data['landmarkManager'][str(i)]:
            landmarks.append((lm['x'], lm['y']))

        landmarks = np.array(landmarks)
        shapes.append(landmarks)
    shapes = np.array(shapes)
    return shapes


if __name__ == '__main__':
    name = '28900.json'
    filename = '/home/ggutierrez/Escritorio/Bisset/export/'+name
    shapes = get_shapes_(filename)
    x, y = shapes[0].T
    x1, y1 = shapes[1].T
    x2, y2 = shapes[2].T
    x3, y3 = shapes[3].T
    #plt.imshow(img, cmap=plt.cm.gray)

    plt.plot(x, -y, 'ro-', ms=1.5, markeredgecolor='k', markerfacecolor='r', mew=0.5)
    plt.plot(x1, -y1, 'yo-', ms=1.5, markeredgecolor='k', markerfacecolor='y', mew=0.5)
    plt.plot(x2, -y2, 'go-', ms=1.5, markeredgecolor='k', markerfacecolor='g', mew=0.5)
    plt.plot(x3, -y3, 'bo-', ms=1.5, markeredgecolor='k', markerfacecolor='b', mew=0.5)

    plt.axis('off')
    #plt.savefig(cfg.resultsFolderPath + 'manual_segmentation_' + name + '.pdf', format='pdf', bbox_inches='tight',
    #            pad_inches=0, dpi=400)
    plt.show()


    # name = 'H18'
    # filename = cfg.path_json + '/' + name + '.json'
    # path = cfg.datasetRoot + '/' + name + '.JPG'
    # img = io.imread(path, as_gray=True)
    # shapes = get_all_landmarks_(filename)
    # subshapes_per_image = []
    # for s in shapes:
    #     subshapes = [s[x:x + cfg.subs_len] for x in range(0, len(s), cfg.subs_len)]
    #     subshapes = np.array(subshapes)
    #     subshapes_per_image.append(subshapes)
    # subshapes_per_image = np.array(subshapes_per_image)
    #
    # x, y = shapes[0].T
    # x1, y1 = shapes[1].T
    # x2, y2 = shapes[2].T
    # x3, y3 = shapes[3].T
    # plt.imshow(img, cmap=plt.cm.gray)
    #
    # plt.plot(x, y, 'ro-', ms=1.5, markeredgecolor='k', markerfacecolor='r', mew=0.5)
    # plt.plot(x1, y1, 'yo-', ms=1.5, markeredgecolor='k', markerfacecolor='y', mew=0.5)
    # plt.plot(x2, y2, 'go-', ms=1.5, markeredgecolor='k', markerfacecolor='g', mew=0.5)
    # plt.plot(x3, y3, 'bo-', ms=1.5, markeredgecolor='k', markerfacecolor='b', mew=0.5)
    #
    # plt.axis('off')
    # plt.savefig(cfg.resultsFolderPath + 'manual_segmentation_' + name + '.pdf', format='pdf', bbox_inches='tight',
    #             pad_inches=0, dpi=400)
    # plt.show()
    # print(len(shapes[0]))
    # print(len(shapes[1]))
    # print(len(shapes[2]))
    # print(len(shapes[3]))
