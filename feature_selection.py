import numpy as np
import numpy.linalg as LA
import random
from scipy.sparse import *
from sklearn.metrics.pairwise import pairwise_distances
from Tools import data_storage as storage
import scipy.io

from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score
from Tools import config as cfg

"""
feature_matrix = np.random.random((100, 5))
displacement_matrix = np.random.randint(0, 50, (6, 5))

d = feature_matrix.shape[0]
d_prime = 70  # dimension a la que se quiere reducir el vector

pairwise_x = []
pairwise_y = []
distances = []

# crear los pares de x & y
for p in range(4):
    (x_i, x_j) = (feature_matrix[:, p], feature_matrix[:, p + 1])
    (y_i, y_j) = (displacement_matrix[:, p], displacement_matrix[:, p + 1])
    x = [x_i, x_j]
    y = [y_i, y_j]
    pairwise_x.append(x)
    pairwise_y.append(y)

    euclidean_dist = LA.norm(y_i - y_j)

    distances.append(euclidean_dist)

d_max = np.max(distances)
t = np.mean(distances)  # el valor de t debe ser analizado
var = np.var(distances)

f_a = []
f_b = []
for dist in distances:
    f_a.append(np.exp(-dist / var))
    if dist <= t:
        f_b.append(0)
    else:
        f_b.append((dist / d_max) ** 2)
print('Similarity:', f_a)
print('Disimilarity:', f_b)

u_a = 0.0
u_b = 0.0

for p in range(len(pairwise_x)):
    x_t = np.asmatrix(pairwise_x[p][0] - pairwise_x[p][1])  # 1x100
    x = x_t.T  # 100x1
    v = np.dot(x, x_t)  # producto matricial #100x100
    u_a += v * f_a[p]  # La dimensión de u_a & u_b es 100x100
    u_b += v * f_b[p]

# Escoger los indices de la matriz de caracteristicas para formar C
index_d_prime = random.sample(range(d), 70)  # indices aleatorios
# indices_sorted = sorted(index_d_prime)
# indices_sorted = np.sort(index_d_prime)

# print(len(index_d_prime))
c_ = []

for p in range(5):
    c_i = []
    feature = feature_matrix[:, p]
    indices = random.sample(range(d), d_prime)
    for index in indices:
        c_i.append(feature[index])
    c_.append(c_i)
print(len(c_))

"""
#for patch in
#for i in indices_sorted:
#    component =
"""

s_c = np.eye(d, d_prime)  # dxd' 100x70

lambda_ = np.trace(np.dot(np.dot(s_c.T, u_b), s_c)) / np.trace(np.dot(np.dot(s_c.T, u_a), s_c))

# sc = np.asmatrix(s_c)
print(s_c.shape)

print(lambda_)

# print(s_c[70,:])
print('UB', u_b.shape)
scores = []
s = s_c[:, 1]
s = np.asmatrix(s)
print('T', s.shape)
ss = np.reshape(s, (1, 100))
print(ss.shape)

for j in range(d_prime):  ## Se podría modificar
    for i in range(d):
        score = np.dot(np.dot(np.reshape(s_c[:, j], (1, d)), (u_b - lambda_ * u_a)), s_c[:, j])
        scores.append(score)
print(len(scores))
"""


def construct_W(X, **kwargs):
    """
    Construct the affinity matrix W through different ways
    Notes
    -----
    if kwargs is null, use the default parameter settings;
    if kwargs is not null, construct the affinity matrix according to parameters in kwargs
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        parameters to construct different affinity matrix W:
        y: {numpy array}, shape (n_samples, 1)
            the true label information needed under the 'supervised' neighbor mode
        metric: {string}
            choices for different distance measures
            'euclidean' - use euclidean distance
            'cosine' - use cosine distance (default)
        neighbor_mode: {string}
            indicates how to construct the graph
            'knn' - put an edge between two nodes if and only if they are among the
                    k nearest neighbors of each other (default)
            'supervised' - put an edge between two nodes if they belong to same class
                    and they are among the k nearest neighbors of each other
        weight_mode: {string}
            indicates how to assign weights for each edge in the graph
            'binary' - 0-1 weighting, every edge receives weight of 1 (default)
            'heat_kernel' - if nodes i and j are connected, put weight W_ij = exp(-norm(x_i - x_j)/2t^2)
                            this weight mode can only be used under 'euclidean' metric and you are required
                            to provide the parameter t
            'cosine' - if nodes i and j are connected, put weight cosine(x_i,x_j).
                        this weight mode can only be used under 'cosine' metric
        k: {int}
            choices for the number of neighbors (default k = 5)
        t: {float}
            parameter for the 'heat_kernel' weight_mode
        fisher_score: {boolean}
            indicates whether to build the affinity matrix in a fisher score way, in which W_ij = 1/n_l if yi = yj = l;
            otherwise W_ij = 0 (default fisher_score = false)
        reliefF: {boolean}
            indicates whether to build the affinity matrix in a reliefF way, NH(x) and NM(x,y) denotes a set of
            k nearest points to x with the same class as x, and a different class (the class y), respectively.
            W_ij = 1 if i = j; W_ij = 1/k if x_j \in NH(x_i); W_ij = -1/(c-1)k if x_j \in NM(x_i, y) (default reliefF = false)
    Output
    ------
    W: {sparse matrix}, shape (n_samples, n_samples)
        output affinity matrix W
    """

    # default metric is 'cosine'
    if 'metric' not in kwargs.keys():
        kwargs['metric'] = 'cosine'

    # default neighbor mode is 'knn' and default neighbor size is 5
    if 'neighbor_mode' not in kwargs.keys():
        kwargs['neighbor_mode'] = 'knn'
    if kwargs['neighbor_mode'] == 'knn' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'y' not in kwargs.keys():
        print ('Warning: label is required in the supervised neighborMode!!!')
        exit(0)

    # default weight mode is 'binary', default t in heat kernel mode is 1
    if 'weight_mode' not in kwargs.keys():
        kwargs['weight_mode'] = 'binary'
    if kwargs['weight_mode'] == 'heat_kernel':
        if kwargs['metric'] != 'euclidean':
            kwargs['metric'] = 'euclidean'
        if 't' not in kwargs.keys():
            kwargs['t'] = 1
    elif kwargs['weight_mode'] == 'cosine':
        if kwargs['metric'] != 'cosine':
            kwargs['metric'] = 'cosine'

    # default fisher_score and reliefF mode are 'false'
    if 'fisher_score' not in kwargs.keys():
        kwargs['fisher_score'] = False
    if 'reliefF' not in kwargs.keys():
        kwargs['reliefF'] = False

    n_samples, n_features = np.shape(X)

    # choose 'knn' neighbor mode
    if kwargs['neighbor_mode'] == 'knn':
        k = kwargs['k']
        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                # compute pairwise euclidean distances
                D = pairwise_distances(X)
                D **= 2
                # sort the distance matrix D in ascending order
                dump = np.sort(D, axis=1)
                idx = np.argsort(D, axis=1)
                # choose the k-nearest neighbors for each instance
                idx_new = idx[:, 0:k+1]
                G = np.zeros((n_samples*(k+1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order='F')
                G[:, 2] = 1
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

            elif kwargs['metric'] == 'cosine':
                # normalize the data first
                X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
                for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
                # compute pairwise cosine distances
                D_cosine = np.dot(X, np.transpose(X))
                # sort the distance matrix D in descending order
                dump = np.sort(-D_cosine, axis=1)
                idx = np.argsort(-D_cosine, axis=1)
                idx_new = idx[:, 0:k+1]
                G = np.zeros((n_samples*(k+1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order='F')
                G[:, 2] = 1
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            t = kwargs['t']
            # compute pairwise euclidean distances
            D = pairwise_distances(X)
            D **= 2
            # sort the distance matrix D in ascending order
            dump = np.sort(D, axis=1)
            idx = np.argsort(D, axis=1)
            idx_new = idx[:, 0:k+1]
            dump_new = dump[:, 0:k+1]
            # compute the pairwise heat kernel distances
            dump_heat_kernel = np.exp(-dump_new/(2*t*t))
            G = np.zeros((n_samples*(k+1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
            G[:, 1] = np.ravel(idx_new, order='F')
            G[:, 2] = np.ravel(dump_heat_kernel, order='F')
            # build the sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W

        elif kwargs['weight_mode'] == 'cosine':
            # normalize the data first
            X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
            for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
            # compute pairwise cosine distances
            D_cosine = np.dot(X, np.transpose(X))
            # sort the distance matrix D in ascending order
            dump = np.sort(-D_cosine, axis=1)
            idx = np.argsort(-D_cosine, axis=1)
            idx_new = idx[:, 0:k+1]
            dump_new = -dump[:, 0:k+1]
            G = np.zeros((n_samples*(k+1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
            G[:, 1] = np.ravel(idx_new, order='F')
            G[:, 2] = np.ravel(dump_new, order='F')
            # build the sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W

    # choose supervised neighborMode
    elif kwargs['neighbor_mode'] == 'supervised':
        k = kwargs['k']
        # get true labels and the number of classes
        y = kwargs['y']
        label = np.unique(y)
        n_classes = np.unique(y).size
        # construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
        if kwargs['fisher_score'] is True:
            W = lil_matrix((n_samples, n_samples))
            for i in range(n_classes):
                class_idx = (y == label[i])
                class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
                W[class_idx_all] = 1.0/np.sum(np.sum(class_idx))
            return W

        # construct the weight matrix W in a reliefF way, NH(x) and NM(x,y) denotes a set of k nearest
        # points to x with the same class as x, a different class (the class y), respectively. W_ij = 1 if i = j;
        # W_ij = 1/k if x_j \in NH(x_i); W_ij = -1/(c-1)k if x_j \in NM(x_i, y)
        if kwargs['reliefF'] is True:
            # when xj in NH(xi)
            G = np.zeros((n_samples*(k+1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                D = pairwise_distances(X[class_idx, :])
                D **= 2
                idx = np.argsort(D, axis=1)
                idx_new = idx[:, 0:k+1]
                n_smp_class = (class_idx[idx_new[:]]).size
                if len(class_idx) <= k:
                    k = len(class_idx) - 1
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class+id_now, 2] = 1.0/k
                id_now += n_smp_class
            W1 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            # when i = j, W_ij = 1
            for i in range(n_samples):
                W1[i, i] = 1
            # when x_j in NM(x_i, y)
            G = np.zeros((n_samples*k*(n_classes - 1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx1 = np.column_stack(np.where(y == label[i]))[:, 0]
                X1 = X[class_idx1, :]
                for j in range(n_classes):
                    if label[j] != label[i]:
                        class_idx2 = np.column_stack(np.where(y == label[j]))[:, 0]
                        X2 = X[class_idx2, :]
                        D = pairwise_distances(X1, X2)
                        idx = np.argsort(D, axis=1)
                        idx_new = idx[:, 0:k]
                        n_smp_class = len(class_idx1)*k
                        G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx1, (k, 1)).reshape(-1)
                        G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx2[idx_new[:]], order='F')
                        G[id_now:n_smp_class+id_now, 2] = -1.0/((n_classes-1)*k)
                        id_now += n_smp_class
            W2 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W2) > W2
            W2 = W2 - W2.multiply(bigger) + np.transpose(W2).multiply(bigger)
            W = W1 + W2
            return W

        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                G = np.zeros((n_samples*(k+1), 3))
                id_now = 0
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    # compute pairwise euclidean distances for instances in class i
                    D = pairwise_distances(X[class_idx, :])
                    D **= 2
                    # sort the distance matrix D in ascending order for instances in class i
                    idx = np.argsort(D, axis=1)
                    idx_new = idx[:, 0:k+1]
                    n_smp_class = len(class_idx)*(k+1)
                    G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                    G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                    G[id_now:n_smp_class+id_now, 2] = 1
                    id_now += n_smp_class
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

            if kwargs['metric'] == 'cosine':
                # normalize the data first
                X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
                for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
                G = np.zeros((n_samples*(k+1), 3))
                id_now = 0
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    # compute pairwise cosine distances for instances in class i
                    D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                    # sort the distance matrix D in descending order for instances in class i
                    idx = np.argsort(-D_cosine, axis=1)
                    idx_new = idx[:, 0:k+1]
                    n_smp_class = len(class_idx)*(k+1)
                    G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                    G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                    G[id_now:n_smp_class+id_now, 2] = 1
                    id_now += n_smp_class
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            G = np.zeros((n_samples*(k+1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                # compute pairwise cosine distances for instances in class i
                D = pairwise_distances(X[class_idx, :])
                D **= 2
                # sort the distance matrix D in ascending order for instances in class i
                dump = np.sort(D, axis=1)
                idx = np.argsort(D, axis=1)
                idx_new = idx[:, 0:k+1]
                dump_new = dump[:, 0:k+1]
                t = kwargs['t']
                # compute pairwise heat kernel distances for instances in class i
                dump_heat_kernel = np.exp(-dump_new/(2*t*t))
                n_smp_class = len(class_idx)*(k+1)
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_heat_kernel, order='F')
                id_now += n_smp_class
            # build the sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W

        elif kwargs['weight_mode'] == 'cosine':
            # normalize the data first
            X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
            for i in range(n_samples):
                X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
            G = np.zeros((n_samples*(k+1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                # compute pairwise cosine distances for instances in class i
                D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                # sort the distance matrix D in descending order for instances in class i
                dump = np.sort(-D_cosine, axis=1)
                idx = np.argsort(-D_cosine, axis=1)
                idx_new = idx[:, 0:k+1]
                dump_new = -dump[:, 0:k+1]
                n_smp_class = len(class_idx)*(k+1)
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_new, order='F')
                id_now += n_smp_class
            # build the sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W



def trace_ratio(X, y, n_selected_features, **kwargs):
    """
    This function implements the trace ratio criterion for feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    n_selected_features: {int}
        number of features to select
    kwargs: {dictionary}
        style: {string}
            style == 'fisher', build between-class matrix and within-class affinity matrix in a fisher score way
            style == 'laplacian', build between-class matrix and within-class affinity matrix in a laplacian score way
        verbose: {boolean}
            True if user want to print out the objective function value in each iteration, False if not
    Output
    ------
    feature_idx: {numpy array}, shape (n_features,)
        the ranked (descending order) feature index based on subset-level score
    feature_score: {numpy array}, shape (n_features,)
        the feature-level score
    subset_score: {float}
        the subset-level score
    Reference
    ---------
    Feiping Nie et al. "Trace Ratio Criterion for Feature Selection." AAAI 2008.
    """

    # if 'style' is not specified, use the fisher score way to built two affinity matrix
    if 'style' not in kwargs.keys():
        kwargs['style'] = 'fisher'
    # get the way to build affinity matrix, 'fisher' or 'laplacian'
    style = kwargs['style']
    n_samples, n_features = X.shape

    # if 'verbose' is not specified, do not output the value of objective function
    if 'verbose' not in kwargs:
        kwargs['verbose'] = False
    verbose = kwargs['verbose']

    if style is 'fisher':
        kwargs_within = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
        # build within class and between class laplacian matrix L_w and L_b
        W_within = construct_W(X, **kwargs_within)
        L_within = np.eye(n_samples) - W_within
        L_tmp = np.eye(n_samples) - np.ones([n_samples, n_samples])/n_samples
        L_between = L_within - L_tmp

    if style is 'laplacian':
        kwargs_within = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        # build within class and between class laplacian matrix L_w and L_b
        W_within = construct_W(X, **kwargs_within)
        D_within = np.diag(np.array(W_within.sum(1))[:, 0])
        L_within = D_within - W_within
        W_between = np.dot(np.dot(D_within, np.ones([n_samples, n_samples])), D_within)/np.sum(D_within)
        D_between = np.diag(np.array(W_between.sum(1)))
        L_between = D_between - W_between

    # build X'*L_within*X and X'*L_between*X
    L_within = (np.transpose(L_within) + L_within)/2
    L_between = (np.transpose(L_between) + L_between)/2
    S_within = np.array(np.dot(np.dot(np.transpose(X), L_within), X))
    S_between = np.array(np.dot(np.dot(np.transpose(X), L_between), X))

    # reflect the within-class or local affinity relationship encoded on graph, Sw = X*Lw*X'
    S_within = (np.transpose(S_within) + S_within)/2
    # reflect the between-class or global affinity relationship encoded on graph, Sb = X*Lb*X'
    S_between = (np.transpose(S_between) + S_between)/2

    # take the absolute values of diagonal
    s_within = np.absolute(S_within.diagonal())
    s_between = np.absolute(S_between.diagonal())
    s_between[s_between == 0] = 1e-14  # this number if from authors' code

    # preprocessing
    fs_idx = np.argsort(np.divide(s_between, s_within), 0)[::-1]
    k = np.sum(s_between[0:n_selected_features])/np.sum(s_within[0:n_selected_features])
    s_within = s_within[fs_idx[0:n_selected_features]]
    s_between = s_between[fs_idx[0:n_selected_features]]

    # iterate util converge
    count = 0
    while True:
        score = np.sort(s_between-k*s_within)[::-1]
        I = np.argsort(s_between-k*s_within)[::-1]
        idx = I[0:n_selected_features]
        old_k = k
        k = np.sum(s_between[idx])/np.sum(s_within[idx])
        if verbose:
            print('obj at iter {0}: {1}'.format(count+1, k))
        count += 1
        if abs(k - old_k) < 1e-3:
            break

    # get feature index, feature-level score and subset-level score
    feature_idx = fs_idx[I]
    feature_score = score
    subset_score = k

    return feature_idx, feature_score, subset_score


def run(features):
    feature_matrix = features.T
    print(feature_matrix.shape)
    labels = np.array(['a', 'b', 'c', 'd', 'e'] * 40)
    print(labels.shape)

    n_feats = 100

    idx, feature_score, subset_score = trace_ratio(feature_matrix, labels, n_feats)
    print(idx)
    print(feature_score)
    print(subset_score)
    selected_features = feature_matrix[:, idx[0:n_feats]]
    print(selected_features.shape)
    return selected_features


def main():
    # load data
    mat = scipy.io.loadmat('/home/ggutierrez/Descargas/COIL20.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features

    # split data into 10 folds
    ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)

    # perform evaluation on classification task
    num_fea = 100    # number of selected features
    clf = svm.LinearSVC()    # linear SVM

    correct = 0
    for train, test in ss:
        # obtain the index of selected features
        idx, feature_score, subset_score = trace_ratio(X[train], y[train], num_fea, style='fisher')

        # obtain the dataset on the selected features
        selected_features = X[:, idx[0:num_fea]]

        # train a classification model with the selected features on the training dataset
        clf.fit(selected_features[train], y[train])

        # predict the class labels of test data
        y_predict = clf.predict(selected_features[test])

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc

    # output the average classification accuracy over all 10 folds
    print('Accuracy:', float(correct)/10)


def select_features(feature_matrix):
    feature_matrix = feature_matrix.T
    # print(feature_matrix.shape)
    labels = np.array(['label'] * cfg.num_of_patches)
    # print(labels.shape)

    n_feats = 100

    idx, feature_score, subset_score = trace_ratio(feature_matrix, labels, n_feats)
    selected_features = feature_matrix[:, idx[0:n_feats]]
    # print(selected_features.shape)

    return selected_features.T



# if __name__ == '__main__':
#     mat = scipy.io.loadmat('/home/ggutierrez/Descargas/COIL20.mat')
#     X = mat['X']  # data
#     X = X.astype(float)
#     y = mat['Y']  # label
#     y = y[:, 0]
#     n_samples, n_features = X.shape
#
#
#     path = '/R_femur/scale_0_25/subshape_16'
#     d, f, c = storage.get_data(path)
#
#     feature_matrix = f[0].T
#     print(feature_matrix.shape)
#     labels = np.array(['a']*40)
#     print(labels.shape)
#
#     n_feats = 100
#
#     idx, feature_score, subset_score = trace_ratio(feature_matrix, labels, n_feats)
#     print(idx)
#     print(feature_score)
#     print(subset_score)
#     selected_features = feature_matrix[:, idx[0:n_feats]]
#     print(selected_features.shape)
#
#     # main()









