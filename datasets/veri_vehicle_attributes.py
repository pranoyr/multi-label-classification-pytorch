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
import xml.etree.ElementTree


def class_mapping():
	class_to_idx = {}
	labels = ['yellow', 'orange', 'green', 'gray' ,'red', 'blue', 'white', 'golden', 'brown', 'black'
            , 'sedan', 'suv','van' , 'hatchback','mpv' , 'pickup', 'bus', 'truck', 'estate' ]
	for i, label in enumerate(labels):
		class_to_idx[label] = i
	return class_to_idx


def make_dataset(csv_file):
	veri_class_to_idx = class_mapping()
	veri_dataset = pd.read_csv(csv_file)
	veri_dataset = veri_dataset.values.tolist()
	return veri_dataset, veri_class_to_idx


def integer_encode(label, class_to_idx):
	""" Convert names to labels.
	"""
	out = []
	for each_label in label:
		each_label = veri_class_to_idx[each_label]
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

	def __init__(self, veri_root_dir, csv_file, num_classes, transform=None):
		'Initialization'
		self.root_dir = veri_root_dir
		self.transform = transform
		self.num_classes = num_classes

		self.data, self.veri_class_to_idx = make_dataset(csv_file)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data)

	def __getitem__(self, index):
		'Generates one sample of data'
		# ---- Get Images ----
		img = Image.open(os.path.join(self.veri_root_dir, self.data[index][0]))
		img = self.transform(img)

		# ---- Get Labels ----
		label = self.data[index][1:4]
		label = integer_encode(label, self.veri_class_to_idx)
		label = one_hot_encode(label, num_classes=self.num_classes)

		target = torch.Tensor(label)
		return img, target
