import os
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
# import shutil
# import random
import numpy as np
# import glob
import tqdm
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn

##Preprocessing and loading data
transform = transforms.Compose([transforms.Resize(800), transforms.CenterCrop(500),
                                transforms.Resize(224), transforms.RandomRotation(10), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

train = datasets.ImageFolder(os.path.join(dir, 'train'), transform=transform)
val = datasets.ImageFolder(os.path.join(dir, 'val'), transform=transform)
test = datasets.ImageFolder(os.path.join(dir, 'test'), transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

##Defining the model
for param in detector.parameters():
  param.requires_grad = False

for p in detector.fc.parameters():
  p.requires_grad = True

classifier = nn.Sequential(nn.Linear(1000, 500),
                           nn.ReLU(),
                           nn.Dropout(.3),
                           nn.Linear(500, 200),
                           nn.ReLU(), nn.Dropout(.3),
                           nn.Linear(200, 30), nn.ReLU(), nn.Dropout(.3),
                           nn.Linear(30, 1))
detector.add_module('classifier',classifier)
