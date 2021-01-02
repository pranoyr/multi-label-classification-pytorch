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


def class_mapping():
	class_to_idx = {}
	labels = ['white', 'silver', 'red', 'black', 'maroon' ,'gray', 'blue', 'grey', 'brown',
			  'purple', 'yellow', 'pink', 'golden', 'orange', 'maruti', 'hyundai', 
			  'renault', 'honda', 'tata', 'mahindra', 'toyota', 'hindustan', 'sedan', 'jeep' ,
			  'bmw', 'volkswagen', 'chevrolet', 'ford', 'fiat', 'nissan', 'mercedes', 'hundai',
			  'datson', 'force', 'datsun', 'jaguar', 'hatchback', 'suv', 'van'] #'unknown',
	for i, label in enumerate(labels):
		class_to_idx[label] = i
	return class_to_idx


def make_dataset(csv_file):
	class_to_idx = class_mapping()
	dataset = pd.read_csv(csv_file)
	dataset = dataset.values.tolist()
	return dataset, class_to_idx


def integer_encode(label, class_to_idx):
	""" Convert names to labels.
	"""
	out = []
	for each_label in label:
		each_label = class_to_idx[each_label]
		out.append(each_label)
	return out


def one_hot_encode(integer_encodings, num_classes):
	""" One hot encode for multi-label classification.
	"""
	onehot_encoded = [0 for _ in range(num_classes)]
	for value in integer_encodings:
		onehot_encoded[value] = 1
	return onehot_encoded


class VehicleAttributes(data.Dataset):
	'Characterizes a dataset for PyTorch'

	def __init__(self, root_dir, csv_file, num_classes, transform=None):
		'Initialization'
		self.root_dir = root_dir
		self.transform = transform
		self.num_classes = num_classes

		self.data, self.class_to_idx = make_dataset(csv_file)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data)

	def __getitem__(self, index):
		'Generates one sample of data'
		# ---- Get Images ----
		img = Image.open(os.path.join(self.root_dir, self.data[index][0]))
		img = self.transform(img)

		# ---- Get Labels ----
		label = self.data[index][1:4]
		label = integer_encode(label, self.class_to_idx)
		label = one_hot_encode(label, num_classes=self.num_classes)

		target = torch.Tensor(label)
		return img, target
