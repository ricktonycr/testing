# Config file

#############################
# DATASET settings
#############################

datasetRoot = 'detection/images'


#############################
# DATABASE settings
#############################

database_root = 'detection/landmark_detection_paper.h5'
database_root_gp = 'detection/gradient_profiling_paper.h5'

# Downscale factor for the pyramid
downScaleFactor = 2
maxLayer = 3
scale_names = ['0_12', '0_25', '0_5', '1']
init_scale = scale_names[0]

# Location to store the Results
resultsFolderPath = 'detection/results'

# Test image path
imagePath = 'detection/images/69.JPG'

testing_folder = 'detection/testing'

##############################
# Training settings
##############################

bone_structures = ['R_femur', 'R_pelvis', 'L_femur', 'L_pelvis']
rigth_side = bone_structures[:2]
left_side = bone_structures[2:]

# Subshapes length
subs_len = 4
# Landmarks for femur and pelvis
num_l_femur = 69
num_l_pelvis = 95

# Patch Settings
num_of_patches = 200
patch_shape = (40, 40)
sample_radius = 30
num_test_patches = 400



#############################
# FEATURE extraction settings
#############################

# Multi-level HOG parameters
# patch size = 40
# Level 1
orientations_1 = 18
pixel_per_cell_1 = (patch_shape[0]/2, patch_shape[0]/2)
cell_per_block_1 = (2, 2)
block_norm_1 = 'L2-Hys'
# Level 2
orientations_2 = 18
pixel_per_cell_2 = (patch_shape[0]/4, patch_shape[0]/4)
cell_per_block_2 = (4, 4)
block_norm_2 = 'L2-Hys'
# normalise = True


##############################
# Active Shape Model parameters
##############################
# Location to store the features

ssm_R_femur = '../Models/SSM_R_femur.model'
ssm_R_pelvis = '../Models/SSM_R_pelvis.model'
ssm_L_femur = '../Models/SSM_L_femur.model'
ssm_L_pelvis = '../Models/SSM_L_pelvis.model'
models = [ssm_R_femur, ssm_R_pelvis, ssm_L_femur, ssm_L_pelvis]

# PCA variance
pca_variance = 0.97


###############################
# Gradient Profiling parameters
###############################
sigma = 1.5
sigma_values = [0, 0.5, 1, 2]  # the first number is not considered
patch_sizes = [0, 10, 20, 30]  # the first number is not considered
testing_patch_sizes = [0, 10, 20, 30]
padding = 50

# Path JSON Files
path_json = 'detection/JSON'


##############################
# Landmark detection settings
##############################

# Feature matching parameters
alpha = 0.05
beta = 0.005

# number of nearest neighbor
s_nn = 5

# Save image settings
dpi = 100
img_format = 'pdf'


# Image Shape
ref_shape = (2345, 3028)
pixel_spacing = 0.139