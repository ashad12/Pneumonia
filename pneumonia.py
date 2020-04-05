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
