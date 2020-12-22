import glob
from Tools import config as cfg
from skimage import io
from Tools import data_storage
import os
from Training import training
import time

def run():
    data_storage.create_database()
    files = glob.glob(cfg.datasetRoot+'/*.png')
    for img in files:
        image = io.imread(img)
        d,f,c = training.training_an_image(image)
        sn = get_name(img)
        print(len(files))
        print(img)
        print(sn)
        print(c)
        s = io.plugin_order()
        print(s)

        data_storage.create_group(sn)


def get_name(path):
    path, fn = os.path.split(path)
    shortname, ext = os.path.splitext(fn)
    return shortname


if __name__ == '__main__':
    # run()
    path = '/R_pelvis/scale_0_5'
    start = time.time()
    d, f, c = data_storage.get_ld_data(path)
    end = time.time()
    print('time', end-start)
    print(d[0].shape)
