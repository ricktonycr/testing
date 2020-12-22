from skimage import io
from Tools import config as cfg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import Refinement.principal_component_analysis as pca_
import pickle
from scipy.spatial import procrustes



data = np.array([(141.33333333333334, 610.6666666666666), (148.0, 604.0), (155.33333333333334, 598.6666666666666),
                (163.33333333333334, 592.0), (169.33333333333334, 584.0), (172.66666666666666, 575.3333333333334),
                (174.66666666666666, 566.6666666666666), (173.5, 556.5), (168.66666666666666, 548.0),
                (165.33333333333334, 540.0), (168.0, 533.0), (172.0, 527.5), (176.0, 521.5), (181.5, 515.5),
                (186.5, 509.5), (190.5, 504.0), (196.5, 499.5), (202.5, 495.0), (209.0, 492.0), (219.0, 493.5),
                (231.0, 495.0), (244.0, 495.5), (254.0, 489.0), (262.5, 480.0), (268.5, 471.5), (274.0, 462.5),
                (275.5, 451.0), (275.5, 438.0), (274.0, 418.5), (273.0, 406.5), (267.5, 396.5), (259.5, 387.5),
                (249.0, 381.0), (238.0, 376.5), (226.0, 373.5), (214.0, 372.5), (200.5, 374.5), (187.5, 378.5),
                (176.5, 384.0), (166.0, 392.5), (160.0, 402.5), (155.5, 414.0), (149.5, 422.5), (138.5, 425.0),
                (126.0, 425.0), (115.5, 424.5), (105.5, 420.0), (99.5, 414.0), (91.5, 405.5), (81.0, 400.0),
                (70.5, 399.5), (61.5, 410.0), (51.0, 420.5), (43.5, 431.0), (38.5, 443.5), (31.0, 456.0), (26.0, 468.0),
                (22.5, 483.0), (25.5, 496.5), (32.5, 507.0), (37.5, 517.5), (41.5, 528.5), (44.5, 540.5), (46.5, 552.0),
                (48.0, 563.0), (49.0, 574.5), (50.0, 585.0), (51.0, 598.0), (51.0, 609.0)])


data2 = np.array([(142.0, 532.0), (145.0, 525.5), (150.5, 520.0), (153.0, 514.0), (156.5, 507.5), (158.5, 500.5),
                (158.5, 493.5), (156.5, 487.5), (152.5, 481.5), (147.33333333333334, 476.0), (151.5, 466.5),
                (154.0, 455.5), (156.5, 445.5), (159.0, 435.0), (163.0, 425.0), (169.0, 415.5), (176.0, 405.5),
                (187.0, 397.5), (199.5, 392.0), (207.5, 389.5), (215.5, 386.5), (221.5, 380.0), (226.0, 374.0),
                (229.0, 366.0), (230.5, 358.0), (231.0, 350.5), (230.5, 342.5), (229.5, 333.5), (222.0, 321.0),
                (215.5, 312.0), (207.0, 306.0), (198.0, 301.0), (188.0, 298.0), (178.5, 297.0), (167.0, 298.0),
                (157.5, 300.5), (149.5, 304.5), (142.0, 309.0), (135.0, 316.0), (129.5, 323.0), (128.0, 332.5),
                (129.0, 342.5), (128.5, 352.5), (122.5, 360.5), (113.0, 367.5), (101.5, 370.5), (88.0, 372.0),
                (78.0, 365.5), (69.0, 359.5), (57.0, 356.0), (47.0, 359.5), (41.0, 369.0), (37.0, 382.5), (34.5, 394.5),
                (30.0, 405.5), (26.0, 416.0), (22.0, 427.5), (23.5, 439.0), (27.0, 451.0), (35.5, 459.5), (41.5, 465.0),
                (46.0, 472.5), (50.5, 480.0), (54.5, 489.5), (57.0, 498.5), (60.5, 506.5), (62.0, 515.5), (64.5, 523.5),
                (66.5, 532.5)])

image = io.imread(cfg.imagePath)
img = io.imread(cfg.imagePath, as_grey=True)

#fig, ax = plt.subplots(1,2)
#ax[0].imshow(image)
#ax[1].imshow(img, cmap=plt.cm.gray)

#plt.show()

##########################
arr = np.random.rand((100))

a = arr.argsort()[:5]

print(a)
##########################


x = np.random.rand(5,5)
print(x)

t = np.trace(x)
print('trace-> ', t)
a = x / t

print(a)


## Testing Nearest Neighbors

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#X = np.random.rand(200,6000).T
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

graph = nbrs.kneighbors_graph(X).toarray()
print(X.shape)
print(distances.shape)

print(graph)



#

file = open(cfg.ssm_femur, 'rb')
pca = pickle.load(file)
shape_ = (69,2)
pca_mean = np.reshape(pca.mean_, shape_)

pca_m_x, pca_m_y = pca_mean.T

model = pca.mean_
x_ = np.reshape(pca.mean_, (138,1))

P = pca.components_
#P = P.T
#b = 0
#b = (-3 * np.sqrt(pca.explained_variance_))
#b = np.reshape(b, (5,1))
data2 -= np.mean(data2, 0)


norm1 = np.linalg.norm(data2)


mtx1, mtx2, dis = procrustes(data2, data2)
#mtx1 *= norm1
new_points = mtx1.flatten()
x = np.reshape(new_points, (138,1))

b = P@(x-x_)
#b = np.reshape((-3 * np.sqrt(pca.explained_variance_),(5,1)))
for i in range(len(pca.explained_variance_)):
    print('range Component {0}: {1:.8f} -- {2:.8f}'.format(i,(-3 * np.sqrt(pca.explained_variance_[i])),(3 * np.sqrt(pca.explained_variance_[i]))))

model_points = np.tile(pca.mean_, (10,1)) + P * b
print('b values:', b)

for i in range(10):
    c_i_a = np.reshape(model_points[i], shape_)

    c_ia_x, c_ia_y = c_i_a.T

    plt.plot(c_ia_x, -c_ia_y, label='Shape')
#plt.show()

plt.plot(pca_m_x, -pca_m_y, label='Mean Shape')
plt.title('MEAN')
plt.legend()
plt.show()

