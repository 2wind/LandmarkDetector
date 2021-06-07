import os
import pandas as pd
import numpy as np
import cv2
import PIL.Image as Image

import math
import random
import imutils

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

class CustomDataSet(Dataset):


    def __init__(self, data_dir: str, image_postfix:str, tsv_postfix:str, landmark_regex:str, landmark_length:int):
        

        self.photo_img_string = image_postfix
        self.photo_tsv_string = tsv_postfix
        self.data_dir = data_dir
        self.landmark_regex = landmark_regex
        self.landmark_length = landmark_length

        files = os.listdir(self.data_dir)

        self.photo_images = [x for x in files if self.photo_img_string in x]
        self.photo_tsvs = [x for x in files if self.photo_tsv_string in x]
        assert(len(self.photo_images) == len(self.photo_tsvs))
        for i in range(len(self.photo_images)):
            x, y = self.photo_images[i], self.photo_tsvs[i]
            assert(os.path.splitext(x)[0] == os.path.splitext(y)[0])

    def __len__(self):
        return len(self.photo_tsvs)

    # load_tsv: load tsv --> return dataframe with name, x, y column.
    def load_tsv(self, name):
        # Loading dataframe
        df = pd.read_csv(os.path.join(self.data_dir, name),  sep='\t')
        df = df.iloc[:99, 0:3]
        
        df.columns = ['name', 'X', 'Y']

        return df


    # load_image: load image --> return plt.Image grayscale.
    def load_image(self, name):
        image = cv2.imread(os.path.join(self.data_dir, name), flags=cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(image)
        return img

    # bounding_box: df(name, x, y) --> return top, left, height, width in integer
    def bounding_box(self, df):
        center = df[df['name'] == '2']

        cy, bottom, cx, right = center['Y'].values[0], df['Y'].max(), center['X'].values[0], df['X'].max()
        dy, dx = bottom - cy, right - cx
        top, left = cy - dy, cx
        # print((left, top), (right, bottom))
        # creating bounding box
        width, height = (right - left), (bottom - top)
        # rand_size_bias = random.uniform(0.9, 1.1)
        # width, height = width * rand_size_bias, height * rand_size_bias

        return int(top), int(left), int(height), int(width)


    def add_random_bias(self, top, left, height, width, bias=0.01):
        top_bias = int(random.uniform(-height*bias,0))
        left_bias = int(random.uniform(-width*bias,0))
        height_bias = -top_bias + int(random.uniform(0, height*bias))
        width_bias = -left_bias + int(random.uniform(0, width*bias))
        top, left, height, width = top + top_bias, left + left_bias, height + height_bias, width + width_bias
        return top, left, height, width

    def extract_landmarks(self, df, landmark_regex, landmark_length):
        # (gathering only needed landmarks)
        df = df.loc[df['name'].str.contains(landmark_regex, regex=True), :]
        # there are **18** landmarks that is unique and valid among all files
        # should we sort df?
        df = df.sort_values(by=['name'])
        df = df.loc[:, ['X', 'Y']]
        df = df.reset_index(drop=True)

        # ... and landmark
        landmark = df.to_numpy(dtype=np.float32)
        return landmark

    def rotate(self, img, landmark, angle):
        angle = random.uniform(-angle, +angle)

        
        transformation_matrix = torch.tensor([
            [+math.cos(math.radians(angle)), -math.sin(math.radians(angle))], 
            [+math.sin(math.radians(angle)), +math.cos(math.radians(angle))]
        ])

        image = imutils.rotate(np.array(img), angle)

        landmark = landmark - 0.5
        new_landmarks = np.matmul(landmark, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def crop(self, img, landmark, top, left, height, width):
        # Cropping image...
        img = TF.crop(img, top, left, height, width)
        #oh, ow = np.array(img).shape[0], np.array(img).shape[1]

        landmark = torch.tensor(landmark) - torch.tensor([[left, top]])
        landmark = landmark / torch.tensor([width, height])
        return img, landmark

    def normalize(self, img, landmark, height, width):
        # normalizing the pixel values
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.6945], [0.33497])


        landmark -= 0.5
        return img, landmark

    def __getitem__(self, index):
        
        img_name = self.photo_images[index]
        tsv_name = self.photo_tsvs[index]

        img = self.load_image(img_name)
        df = self.load_tsv(tsv_name)

        top, left, height, width = self.bounding_box(df)
        top, left, height, width = self.add_random_bias(top, left, height, width, 0.02)
        landmark = self.extract_landmarks(df, self.landmark_regex, self.landmark_length)

        # rand_top = int(top) + random.randint(-int(height * 0.1), int(height * 0.1))
        # rand_left = int(left) + random.randint(0, int(width * 0.2))

        img, landmark = self.crop(img, landmark, top, left, height, width)

        # resizing image..
        img = TF.resize(img, (224, 224))
        # packing image
        # use dsplit when RGB to make 224x224x3 --> 3x224x224
        #img = np.dsplit(img, img.shape[-1])

        img, landmark = self.rotate(img, landmark, 5)
        img, landmark = self.normalize(img, landmark, height, width)

        #arr = arr.flatten('F')

        return img, landmark, img_name