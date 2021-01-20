import os
from pickle import dump
import torch
import cv2
from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import pickle as pk

def integer_encode(label, word2int):
    """ Convert names to labels.
    """
    out = []
    print(out)
    for each_label in label:
        each_label = word2int[each_label]
        out.append(each_label)
    return out
    
def one_hot_encode(integer_encodings, num_classes):
    """ One hot encode for multi-label classification.
    """
    onehot_encoded = [0 for _ in range(num_classes)]
    for value in integer_encodings:
        onehot_encoded[value] = 1

    return onehot_encoded


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root, anns_file_path, transform=None):
        'Initialization'
        self.root = root
        self.transform = transform

        with open(anns_file_path, 'rb') as f:
            self.anns_file = pk.load(f)
        
        self.word2int = {'Hatchback':0, 'MiniVan':1, 'Truck-Box-Large':2, 'Truck-Pickup':3, 'Sedan':4, 
        'Truck-Flatbed':5,'Bus':6, 'Truck-Box-Med':7, 'Taxi':8, 'Police':9, 'Suv':10, 'Van':11, 'Truck-Util':12,
        'Yellow':13, 'Multi':14, 'Black':15, 'Blue':16, 'Red':17, 'Green':18, 'Gray':19, 'Silver':20, 'Beige':21, 
        'Brown':22,'White':23, 'Orange':24}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.anns_file)
      
    def __getitem__(self, index):
        'Generates one sample of data'

        # ---- Get Images ----
        img = Image.open(os.path.join(self.root, self.anns_file[index]['filename'], self.anns_file[index]['frame_no']+'.jpg'))
        # get bounding box
        x1, y1, x2, y2 = self.anns_file[index]['bounding_boxes']
        x1 = int(float(x1))
        y1 = int(float(y1))
        x2 = int(float(x2))
        y2 = int(float(y2))
        # crop image
        img = np.array(img)
        crop_img = img[y1:y2, x1:x2]
        img = Image.fromarray(crop_img)
        img = self.transform(img)
        
        # ---- Get Labels ----
        label = self.anns_file[index]['labels']
        label_int_encoded = integer_encode(label, self.word2int)
        label = one_hot_encode(label_int_encoded, num_classes=25)

        target = torch.Tensor(label)
        target_int_encoded = torch.Tensor(label_int_encoded)
        return img, target, target_int_encoded
     
