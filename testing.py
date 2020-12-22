from skimage import io
from Tools import config as cfg, pyramid_gaussian, sample_patches
from Testing import landmark_detection
import extract_features
import pydicom
import matplotlib.pyplot as plt


def run():
    image = io.imread(cfg.imagePath, as_grey=True)
    image_pyramid = pyramid_gaussian.get_pyramid(image)
    for img in image_pyramid[::-1]:
        cfg.num_of_patches = cfg.num_test_patches
        patches, centres = sample_patches.create_patches_randomly(img)
        F = extract_features.extractFeaturesForPatches(patches)
        landmark_detection.run()



def read_dicom():
    path = '/home/ggutierrez/Images/hacked/file.dcm'

    image = pydicom.dcmread(path)
    print('Body Part Examined: ', image.BodyPartExamined)
    print('Image Shape: ', image.pixel_array.shape)
    print('Pixel Spacing: ', image.PixelSpacing)
    print('Patient Age: ', image.PatientAge)

    new_spacing = ['0.1', '0.1']

    image.PixelSpacing = new_spacing

    print('NEW PIXEL SPACING: ', image.PixelSpacing)
    print('Image Shape 2: ', image.pixel_array.shape)
    data = image.pixel_array
    plt.imshow(data)
    plt.show()

    data = data[1000:1700,200:1000]
    print(data)
    print(data.shape)
    plt.imshow(data)
    plt.show()

    ## save image modified

    image.save_as('new_dicom.dcm')
    img = pydicom.dcmread('new_dicom.dcm')
    print('NEW PIXEL SPACING 2', img.PixelSpacing)


if __name__ == '__main__':
    #run()
    read_dicom()

