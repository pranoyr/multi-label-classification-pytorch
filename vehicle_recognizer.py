import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import random
import numpy as np
from datasets.vehicle_attributes import class_mapping
from PIL import Image
import cv2

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class VehicleRecognizer:
	""" Vehicle make/colour/type 
	"""

	def __init__(self):
		# CUDA for PyTorch
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")

		self.veh_colour = ['white', 'silver', 'red', 'black', 'maroon' ,'gray' ,'blue' ,'grey' ,'brown',
			  'purple', 'yellow', 'pink' ,'golden' ,'orange']
		self.veh_make = ['maruti' ,'hyundai', 'renault', 'honda' ,'tata',
			  'mahindra', 'toyota' ,'hindustan', 'sedan' ,'jeep' ,'bmw' ,'volkswagen',
			  'chevrolet' ,'ford', 'fiat' ,'nissan' ,'mercedes' , 'hundai',
			  'datson', 'force', 'datsun', 'jaguar']
		self.veh_model = ['hatchback' , 'suv' , 'van']

		# loading class mapping
		class_to_idx = class_mapping()
		self.idx_to_class = {}
		for name, label in class_to_idx.items():
			self.idx_to_class[label] = name
	
		self.transform = transforms.Compose([
			transforms.Resize((150, 150)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
				0.229, 0.224, 0.225])
		])

		# define model
		self.model = resnet18(num_classes=40)
		# load weights
		checkpoint = torch.load('/Volumes/Seagate/Neuroplex/vehicle_model_classifier.pth', map_location='cpu')
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.model = self.model.to(self.device)
		self.model.eval()

	def detect(self, img, bbox):
		""" 
        Args
                img      : Image of shape (None,None,None).
                bbs      : 1D list of bounding box.
        """
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		img = Image.fromarray(img)
		
		img = self.transform(img)
		img = torch.unsqueeze(img, dim=0)
		data = {}
		with torch.no_grad():
			outputs =self.model(img)
			outputs = torch.sigmoid(outputs)
			scores, indices = torch.topk(outputs, dim=1, k=40)
			mask = scores > 0.4
			preds = indices[mask]

			preds = [self.idx_to_class[label.item()] for label in preds]

			for pred in preds:
				if(pred in self.veh_colour):
					data.update({'colour':pred})
				if(pred in self.veh_make):
					data.update({'make':pred})
				if(pred in self.veh_model):
					data.update({'model':pred})
		return data
	
veh = VehicleRecognizer()
img = cv2.imread("./images/car_model.jpg")
results = veh.detect(img,[100,50,100,50])
print(results)