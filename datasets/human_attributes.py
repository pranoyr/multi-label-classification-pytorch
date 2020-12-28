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
import pandas as pd


def integer_encode(img_label, class_to_idx):
    encoded_label = [class_to_idx[label] for label in img_label.split()] 
    return encoded_label


def create_dict_from_labels(label_file):
    """ Create dictionary from Labes.txt
    """
    dict_labels = {}
    for i in label_file.split('\n'):
        if i:
            text = i.split()
            label_text = ' '.join(text[1:])
            dict_labels.update({text[0]:label_text})
    return dict_labels

def make_dataset(dataset_path):
    """ Loading the img_ids and labels.
    """
    img_paths = []
    labels = []
    class_to_idx = class_mapping()
    dataset_path = dataset_path + '/PETA'
    for folder in os.listdir(dataset_path):
        # creating dictionary from Label.txt
        label_file = open(os.path.join(dataset_path, folder, 'Label.txt'),'r').read()
        dict_labels = create_dict_from_labels(label_file)
        # creating img_paths and labels
        for filename in os.listdir(os.path.join(dataset_path, folder)):
            final_path = os.path.join(dataset_path, folder, filename)
            if final_path.split('.')[1]!='txt':
                img_paths.append(final_path)
                id = filename.split('_')[0]
                label = dict_labels[id]
                label = integer_encode(label.lower(), class_to_idx)
                labels.append(label)

    return img_paths, labels

def class_mapping():
    class_to_idx = {}
    all_labels = open('labels.txt').read()
    all_labels = all_labels.split()

    for color in ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']:
        all_labels.append('footwear'+color)
        all_labels.append('hair'+color)
        all_labels.append('lowerbody'+color)
        all_labels.append('upperbody'+color)
    for i, label in enumerate(all_labels):
        class_to_idx.update({label.lower():i})
    return class_to_idx

def one_hot_encode(integer_encodings, num_classes):
    """ One hot encode for multi-label classification.
    """
    onehot_encoded = [0 for _ in range(num_classes)]
    for value in integer_encodings:
        onehot_encoded[value] = 1
    return onehot_encoded


class HumanAttributes(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dataset_path, num_classes, transform=None):
        'Initialization'
        self.transform = transform
        self.img_paths, self.labels = make_dataset(dataset_path)
        self.num_classes = num_classes

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # ---- Get Images ----
        img = Image.open(self.img_paths[index])
        img = self.transform(img)

        # ---- Get Labels ----
        label = self.labels[index]
        label = one_hot_encode(label, num_classes=self.num_classes)
        target = torch.Tensor(label)

        return img, target
