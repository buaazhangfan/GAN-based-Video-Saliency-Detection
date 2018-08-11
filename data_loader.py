import glob
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from constants import *

class DataLoader(object):

    def __init__(self, batch_size = 5):
        #reading data list
        self.list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToResizedImagesTrain, '*train*'))]
        self.batch_size = batch_size
        self.size = len(self.list_img)
        self.cursor = 0
        self.num_batches = self.size / batch_size

    def get_batch(self): # Returns 
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)
            
        img = torch.zeros(self.batch_size, 3, 192, 256)
        sal_map = torch.zeros(self.batch_size, 1, 192, 256)
        
        to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.

        for idx in range(self.batch_size):
            curr_file = self.list_img[self.cursor]
            full_img_path = os.path.join(pathToResizedImagesTrain, curr_file + '.png')
            full_map_path = os.path.join(pathToResizedMapsTrain, curr_file + '.png')
            self.cursor += 1
            inputimage = cv2.imread(full_img_path) # (192,256,3)
            img[idx] = to_tensor(inputimage)
            
            saliencyimage = cv2.imread(full_map_path,0)
            saliencyimage = np.expand_dims(saliencyimage,axis=2)
            sal_map[idx] = to_tensor(saliencyimage)
            
        return (img,sal_map)
