import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
import argparse
import tensorboardX
import cv2
import os
import random
import numpy as np
from train import train_epoch
from torch.nn import BCEWithLogitsLoss
from validation import val_epoch
from opts import parse_opts
from torch.optim import lr_scheduler
from dataset import get_training_set, get_validation_set
from datasets.veri_vehicle_attributes import class_mapping
from PIL import Image



def main():
    opt = parse_opts()
    print(opt)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{opt.gpu}" if use_cuda else "cpu")

    class_to_idx = class_mapping()
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
   
    transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=3),
        transforms.Resize((opt.img_H, opt.img_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    # define model
    model = resnet18(num_classes=opt.num_classes)
    # load weights
    checkpoint = torch.load('/Users/pranoyr/Downloads/model100.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # checkpoint = torch.load('/Users/pranoyr/Desktop/weights/vehicle_classifier.pth', map_location='cpu')
    # model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    #print(class_to_idx)
    img = cv2.imread('/Users/pranoyr/Desktop/reid/c2.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    # img = img.to(device)
    # img = Image.open('./images/sample6.jpg')
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    with torch.no_grad():
        outputs = model(img)
        outputs = torch.sigmoid(outputs)
        scores, indices = torch.topk(outputs, dim=1, k=2)
        mask = scores > 0.5
        preds = indices[mask]
    
        preds = [idx_to_class[label.item()] for label in preds]
        print(preds)

if __name__ == "__main__":
    main()
