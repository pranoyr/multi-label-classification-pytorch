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
import xml.etree.ElementTree as ET

def read_xml(xml_file):
    color_class_to_idx = {}
    make_class_to_idx = {}
    colorlabels = ['yellow', 'orange', 'green', 'gray' ,'red', 'blue', 'white', 'golden', 'brown', 'black']
    makelabels  = ['sedan', 'suv', 'van', 'hatchback', 'mpv' , 'pickup', 'bus', 'truck', 'estate']

    for i, label in enumerate(colorlabels):
        color_class_to_idx[i + 1] = label

    for i, label in enumerate(makelabels):
        make_class_to_idx[i + 1] = label

    with open(xml_file,'r') as f:
        tree = ET.fromstring(f.read())
    data = []
    for Item in tree.iter('Item'):
        df = [Item.attrib['imageName'], color_class_to_idx[int(Item.attrib['colorID'])], make_class_to_idx[int(Item.attrib['typeID'])]]
        data.append(df)
    return data


def class_mapping():
    class_to_idx = {}
    labels = ['yellow', 'orange', 'green', 'gray' ,'red', 'blue', 'white', 'golden', 'brown', 'black',
			  'sedan', 'suv', 'van', 'hatchback', 'mpv' , 'pickup', 'bus', 'truck', 'estate']

    for i, label in enumerate(labels):
        class_to_idx[label] = i
    return class_to_idx


def make_dataset(xml_file):
	class_to_idx = class_mapping()
	veri_dataset = pd.read_xml(xml_file)
	return veri_dataset, class_to_idx


def integer_encode(label, class_to_idx):
	""" Convert names to labels.
	"""
	out = []
	for each_label in label:
		each_label = class_to_idx[each_label]
		veri_out.append(each_label)
	return veri_out


def one_hot_encode(integer_encodings, num_classes):
	""" One hot encode for multi-label classification.
	"""
	veri_onehot_encoded = [0 for _ in range(num_classes)]
	for value in integer_encodings:
		veri_onehot_encoded[value] = 1
	return veri_onehot_encoded


class VeriVehicleAttributes(data.Dataset):
	'Characterizes a dataset for PyTorch'

	def __init__(self, root_dir, xml_file, num_classes, transform=None):
		'Initialization'
		self.root_dir = root_dir
		self.transform = transform
		self.num_classes = num_classes

		self.data, self.class_to_idx = make_dataset(xml_file)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data)

	def __getitem__(self, index):
		'Generates one sample of data'
		# ---- Get Images ----
		img = Image.open(os.path.join(self.root_dir, self.data[index][0]))
		img = self.transform(img)

		# ---- Get Labels ----
		label = self.data[index][1:3]
		label = integer_encode(label, self.class_to_idx)
		label = one_hot_encode(label, num_classes=self.num_classes)

		target = torch.Tensor(label)
		return img, target
