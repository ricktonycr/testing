import numpy as np
import tables

from Tools import config as cfg


def create_database(db_path=cfg.database_root):
    database = tables.open_file(db_path, mode='w', title='Training Data')
    database.close()


def create_group(db_path=cfg.database_root, name='name', parent=None):
    database = tables.open_file(db_path, mode='a', title='Training Data')
    if parent != None:
        g = database.create_group(parent, name, name)

    else:
        g = database.create_group('/', name, name)

    return g
    # database.close()


def save_data(group, name, data, db_path=cfg.database_root):
    database = tables.open_file(db_path, mode='a', title='Training Data')
    database.create_array(group, name, obj=data)
    # database.create_carray(group, name, shape=(69,40,40,3), obj=data)
    database.close()


def read_data(db_path=cfg.database_root):
    database = tables.open_file(db_path, mode='r')
    loc = database.root.H1.scale_1  # ver la forma para leer todos los datos
    data = database.walk_nodes(loc)
    print(loc)
    for leaf in loc:
        print(leaf.name, '->', leaf.read().shape, type(leaf.read()))

    database.close()


# Get data for landmark detection
def get_data(path):
    db = tables.open_file(cfg.database_root, mode='r')

    d = []
    f = []
    c = []
    # list all the image names
    for group in db.root:
        img_name = group.__str__().split()[0]  # get the name
        new_path = img_name + path
        if group.__contains__(new_path.split('/')[2]):
            for leaf in db.list_nodes(new_path):
                name = leaf.name[0]
                if name == 'D':
                    d.append(leaf.read())

                elif name == 'F':
                    f.append(leaf.read())

                else:
                    c.append(leaf.read())

    db.close()
    print('READED...')
    D = []
    F = []
    C = []

    for d_, f_, c_ in zip(d, f, c):
        indexes = np.random.choice(c_.shape[1], 40, replace=False)  # selecting
        patches_d = []
        patches_f = []
        patches_c = []

        for id in indexes:
            p_d = d_[:, id]
            patches_d.append(p_d)
            patches_f.append(f_[:, id])
            patches_c.append(c_[:, id])

        patches_d = np.array(patches_d).T
        patches_f = np.array(patches_f).T
        patches_c = np.array(patches_c).T

        D.append(patches_d)
        F.append(patches_f)
        C.append(patches_c)

    return D, F, C


# Get data for gradient Profiling
def get_patches(path, n_landmark):
    db = tables.open_file(cfg.database_root_gp, mode='r')
    training_patches = []
    for group in db.root:
        g_name = group.__str__().split()[0]
        new_path = g_name + path

        if group.__contains__(new_path.split('/')[2]):
            for leaf in db.list_nodes(new_path):
                if leaf.name[6::] == str(n_landmark):
                    patch = leaf.read()
                    training_patches.append(patch)
    db.close()
    print('READED...')
    return training_patches


# Get data for gradient Profiling
def get_patches_(path, n_landmark=0):
    db = tables.open_file(cfg.database_root_gp, mode='r')
    training_patches = []
    for group in db.root:
        g_name = group.__str__().split()[0]
        new_path = g_name + path

        if group.__contains__(new_path.split('/')[2]):
            for leaf in db.list_nodes(new_path):
                training_patches.append(leaf.read())
    db.close()
    print('READED GP...')
    return training_patches  # list of matrices with all the patches for a scale


# Read Landmark Detection data modified
def get_ld_data(path):
    db = tables.open_file(cfg.database_root, mode='r')

    d_per_image = []
    f_per_image = []
    c_per_image = []

    # list all the image names
    for group in db.root:
        d = []
        f = []
        c = []

        img_name = group.__str__().split()[0]  # get the name
        new_path = img_name + path

        if group.__contains__(new_path.split('/')[2]):  # if contains bone
            for subs in db.list_nodes(new_path):
                idx_subs = int(subs.__str__().split(' ')[-1][10:-1])

                for leaf in subs:
                    name = leaf.name[0]
                    if name == 'D':
                        d.append((idx_subs, leaf.read()))

                    elif name == 'F':
                        f.append((idx_subs, leaf.read()))

                    else:
                        c.append((idx_subs, leaf.read()))

            d.sort()
            f.sort()
            c.sort()

            d = [i[1] for i in d]
            f = [i[1] for i in f]
            c = [i[1] for i in c]

            d_per_image.append(d)
            f_per_image.append(f)
            c_per_image.append(c)

    db.close()

    d_per_image = np.array(d_per_image)
    f_per_image = np.array(f_per_image)
    c_per_image = np.array(c_per_image)

    D_ = []
    F_ = []
    C_ = []
    for i in range(d_per_image.shape[1]):  # for each subshape get the random patches
        D = []
        F = []
        C = []
        for d_, f_, c_ in zip(d_per_image[:, i], f_per_image[:, i], c_per_image[:, i]):
            indexes = np.random.choice(c_.shape[1], 40, replace=False)  # selecting
            patches_d = []
            patches_f = []
            patches_c = []

            for id in indexes:
                patches_d.append(d_[:, id])
                patches_f.append(f_[:, id])
                patches_c.append(c_[:, id])

            patches_d = np.array(patches_d).T
            patches_f = np.array(patches_f).T
            patches_c = np.array(patches_c).T

            D.append(patches_d)
            F.append(patches_f)
            C.append(patches_c)
        D_.append(D)
        F_.append(F)
        C_.append(C)

    for i in range(len(D_)):
        D_[i] = np.concatenate(D_[i], axis=1)
        F_[i] = np.concatenate(F_[i], axis=1)
        C_[i] = np.concatenate(C_[i], axis=1)

    return D_, F_, C_


def select_patches(d, f, c):
    D = []
    F = []
    C = []

    for d_, f_, c_ in zip(d, f, c):
        indexes = np.random.choice(c_.shape[1], 40, replace=False)  # selecting
        patches_d = []
        patches_f = []
        patches_c = []

        for id in indexes:
            patches_d.append(d_[:, id])
            patches_f.append(f_[:, id])
            patches_c.append(c_[:, id])

        patches_d = np.array(patches_d).T
        patches_f = np.array(patches_f).T
        patches_c = np.array(patches_c).T

        D.append(patches_d)
        F.append(patches_f)
        C.append(patches_c)

    return D, F, C
