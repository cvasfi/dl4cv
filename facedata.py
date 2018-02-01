import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import cv2 as cv


class FaceData(data.Dataset):

    def __init__(self, dataset_csv="data/fer2013.csv", dataset_type='Training', transform=None):
        self.root_dir_name = os.path.dirname(dataset_csv)
        df = pd.read_csv(dataset_csv)
        #self.facial_expressions = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
        self.images = df.loc[df['Usage'] == dataset_type, ['pixels']]
        #print(self.images)
        #print("**********")
        self.images = self.transform_images()
        self.labels = df.loc[df['Usage'] == dataset_type, ['emotion']]
        self.labels = self.transform_labels()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # mode 'L': 8-bit pixels, black and white
        #image = Image.fromarray(image, 'L')

        if self.transform:
            image = np.squeeze(image,0)
            # mode 'L': 8-bit pixels, black and white
            image = Image.fromarray(np.uint8(image))
            #filename = "../images/imageB" + str(idx) + ".jpeg"
            #image.save(filename)
            image = self.transform(image)

        return image, label


    def transform_images(self):
        """
        :param data_frame: panda data frame
        :return: data_frame: numpy array
        """
        data_frame = self.images['pixels']  # Selecting Pixels Only
        data_frame = data_frame.values  # Converting from Panda Series to Numpy Ndarray
        data_frame = data_frame.reshape((data_frame.shape[0], 1))  # Reshape for the subsequent operation
        # convert pixels from string to ndarray
        data_frame = np.apply_along_axis(lambda x: np.array(x[0].split()).astype(dtype=float), 1, data_frame)
        #print(data_frame.shape)
        data_frame = data_frame.reshape((data_frame.shape[0], 1, 48, 48))  # reshape to NxHxWxC

        print("######### Mean pxiels ###########")
        mean_pixel = np.mean(data_frame, axis=(1,2,3))
        std_pixel = np.std(data_frame, axis=(1,2, 3))
        print(mean_pixel)
        print("######### Std pxiels ###########")
        print(std_pixel)

        for i in xrange(data_frame.shape[0]):
            data_frame[i, :] = data_frame[i, :] - (mean_pixel[i] * (3.125 / std_pixel[i]))

        print("######### Mean pxiels ###########")
        mean_pixel = np.mean(data_frame, axis=(1, 2, 3))
        std_pixel = np.std(data_frame, axis=(1, 2, 3))
        print(mean_pixel)
        print("######### Std pxiels ###########")
        print(std_pixel)

        #print("######### Images ##########")
        #print(data_frame)

        data_frame = (data_frame - np.nanmean(data_frame, axis=0)) / np.nanstd(data_frame, axis=0)

        print("######### Mean Image ##########")
        mean_image = np.nanmean(data_frame, axis=0)
        std_image = np.nanstd(data_frame, axis=0)

        print(mean_image)
        print(std_image)

        return data_frame

    def transform_labels(self):
        """
        :param data_frame: panda data frame with target columns
        :return: data_frame: Numpy array of shape (N * number of classes)
        """
        data_frame = self.labels['emotion']  # Selecting Emotion Only
        data_frame = data_frame.astype('category', categories=list(range(7)))
        #data_frame = pd.get_dummies(data_frame)
        data_frame = data_frame.values
        return data_frame



