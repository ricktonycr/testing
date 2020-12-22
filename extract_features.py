import os
import pickle
from skimage import io, util
from Tools import feature_extractor
import numpy as np


def run(inputPath, outputPath):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    print('Extracting features from ' + inputPath + ' in ' + outputPath)

    extractAndStoreFeatures(inputPath, outputPath)


def extractAndStoreFeatures(inputFolder, outputFolder):
    fileList = os.listdir(inputFolder)
    imagesList = filter(lambda element: '.png' in element, fileList)

    for filename in imagesList:
        imagePath = inputFolder + '/' + filename
        outputPath = outputFolder + '/' + filename + '.feat'

        print('Extracting features for ' + imagePath)

        image = io.imread(imagePath, as_grey=True)
        image = util.img_as_uint(image)
        feats = feature_extractor.extractHOGFeatures(image)

        outputFile = open(outputPath, 'wb')
        pickle.dump(feats, outputFile)
        outputFile.close()


def extractFeaturesForPatches(patches):
    features = []
    for patch in patches:
        feats = extract_feature_vector(patch)
        features.append(feats)

    # Convert list to ndarray
    features = np.array(features).transpose()

    return features


def extract_feature_vector(image):
    #image = util.img_as_uint(image)  # doubt ##image = util.img_as_float(image)
    feats = feature_extractor.extractHOGFeatures(image)
    return feats


if __name__ == '__main__':
    inputPath = '/home/ggutierrez/Test/input'
    outputPath = '/home/ggutierrez/Test/output'
    run(inputPath, outputPath)