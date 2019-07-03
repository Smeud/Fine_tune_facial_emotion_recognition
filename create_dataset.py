# USAGE
# python3 create_dataset.py

# import the necessary packages
from src import config_params
from imutils import paths
import shutil
import os
import cv2 as cv

# loop over the data splits
for split in (config_params.TRAIN, config_params.TEST, config_params.VAL):
    # grab all image paths in the current split
    print("[INFO] processing '{} split'...".format(split))
    p = os.path.sep.join([config_params.ORIGIN_DT_PATH, split])
    imagePaths = list(paths.list_images(p))

    # loop over the image paths
    for imagePath in imagePaths:

        # load the image, swap color channels, and resize it to be a fixed
        # 128x128 pixels while ignoring aspect ratio
        filename = cv.imread(imagePath)
        filename = cv.cvtColor(filename, cv.COLOR_BGR2RGB)
        filename = cv.resize(filename, (128, 128))

        # extract class label from the filename
        filename = imagePath.split(os.path.sep)[-1]
        label = config_params.CLASSES[int(filename.split("_")[0])]



        # construct the path to the output directory
        dirPath = os.path.sep.join([config_params.BASE_DT_PATH, split, label])

        # if the output directory does not exist, create it
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # construct the path to the output image file and copy it
        p = os.path.sep.join([dirPath, filename])
        shutil.copy2(imagePath, p)
