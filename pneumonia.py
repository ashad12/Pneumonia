import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

##Preprocessing and loading data
transform = transforms.Compose([#transforms.Resize(800), transforms.CenterCrop(500),
                                transforms.Resize(224), transforms.RandomRotation(10), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

train = datasets.ImageFolder(os.path.join(dir, 'train'), transform=transform)
val = datasets.ImageFolder(os.path.join(dir, 'val'), transform=transform)
test = datasets.ImageFolder(os.path.join(dir, 'test'), transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=len(test.targets), shuffle=True)

##Defining the model
detector = models.resnet50(pretrained=True)
for param in detector.parameters():
  param.requires_grad = False

for p in detector.fc.parameters():
  p.requires_grad = True
for p in detector.avgpool.parameters():
  p.requires_grad = True

classifier = nn.Sequential(nn.Linear(1000, 500),
                           nn.ReLU(),
                           nn.Dropout(.3),
                           nn.Linear(500, 200),
                           nn.ReLU(), nn.Dropout(.3),
                           nn.Linear(200, 30), nn.ReLU(), nn.Dropout(.3),
                           nn.Linear(30, 1))
detector.add_module('fc',classifier)

## Optimization and loss criteria
import torch.optim as optim
params = list(detector.avgpool.parameters()) + list(detector.fc.parameters())
optimizer = optim.Adam(params, lr=.001, betas=(.4, .999))
criteria = nn.BCEWithLogitsLoss()

## Training function:
def train(model, epoch, data_loader, valid_loader, optimizer,
          criterion, save_model, early_stop=.1):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  val_loss_min = np.inf
  model = model.to(device)

  for e in range(epoch):
    train_loss = 0
    val_loss = 0

    model.train()
    print('---------------------------\nTRAINING PHASE EPOCH %s:'%(e+1))
    for data, target in tqdm.notebook.tqdm(data_loader):
      data = data.to(device)
      target = target.float().to(device)

      pred = model.forward(data)
      loss = criterion(pred.squeeze(), target)
      train_loss += loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    train_loss = train_loss/len(data_loader)

    model.eval()
    print('VALIDATING PHASE EPOCH %s:'%(e+1))
    for data, target in tqdm.notebook.tqdm(valid_loader):
      data = data.to(device)
      target = target.float().to(device)
      pred = model.forward(data)
      loss = criterion(pred.squeeze(), target)
      val_loss += loss

    val_loss = val_loss/len(val_loader)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        (e+1), train_loss, val_loss))

    ## save the model if validation loss has decreased
    if val_loss < val_loss_min:
      delta = val_loss_min-val_loss
      print('#########################\nValidation loss decreased by %.6f (%.3f)%%\
      Saving model....\n#########################'%(delta, delta/val_loss_min*100))
      torch.save(model.state_dict(), save_model)

      val_loss_min = val_loss
      terminate = 0

    elif val_loss >= (1+early_stop)*val_loss_min:
      terminate +=1

    else:
      terminate =0

    if terminate == 5:
      print('it is diverging...')
      break
    # return trained model
  return model

pn_detector = train(detector, 100, train_loader, val_loader, optimizer,
                    criterion, os.path.join(dir, 'model_params.pt'))

## Testing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pn_detector.load_state_dict(torch.load(os.path.join(dir, 'model_params.pt')))

data,target = next(iter(test_loader2))
data = data.to(device)
pred = pn_detector(data)
pred = torch.round(F.sigmoid(pred.squeeze()))
pred = pred.cpu().detach().numpy()

T = target.numpy()

rec = recall_score(T, pred)
per = precision_score(T, pred)
cm = confusion_matrix(T, pred)
print(f'accuracy: {acc}\nrecal: {rec} \npercision: {per}')
print(f'\nCM:\n{cm}')
