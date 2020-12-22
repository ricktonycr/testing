from skimage.util.shape import view_as_windows
from skimage import io, data, color, util
from Tools import feature_extractor, config as cfg
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open(cfg.imagePath)
image2 = io.imread(cfg.imagePath)
#image2 = color.rgb2gray(image2)


area = (50, 300, 200, 500)

cropped_img = image.crop(area)
patch = util.img_as_float(cropped_img)
gray_img = color.rgb2gray(patch)
#cropped_img.show()

cropped_image = image2[300:500, 50:200]
cropped_image = color.rgb2gray(cropped_image)

#patch = util.img_as_float(cropped_image)
feats1 = feature_extractor.extractHOGFeatures(gray_img)
feats2 = feature_extractor.extractHOGFeatures(cropped_image)

patch_shape = (40, 40)



fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(image)
ax1.set_title('Original Image')
#ax2.imshow(patch1[100][100])
ax2.imshow(cropped_image)
ax2.set_title('skimage')
ax3.imshow(gray_img)
ax3.set_title('PIL')

plt.show()

