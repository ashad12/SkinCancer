import numpy as np
import pandas as pd
import torch
import os
import glob
import shutil
import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from google.colab import drive
# drive.mount('/content/gdrive')

dir = '/content/gdrive/My Drive/Colab Notebooks/Skin Cancer/'
orig_data_dir = dir + 'Original data'

### Preprocessing: sorting data based on lesion type
id2type = pd.read_csv(dir + 'HAM10000_metadata.csv')
types = id2type['dx'].unique().astype(str)

dataset_dir = dir + 'Dataset/'
if not os.path.isdir(dataset_dir):
  os.makedirs(dataset_dir)

for t in types:
  type_dir = dataset_dir + t
  if not os.path.isdir(type_dir):
    os.makedirs(type_dir)

## following code is run only ONCE to sort out data based on lesion type
for file in tqdm.notebook.tqdm(glob.glob(orig_data_dir +'/*/*.jpg')):
  file_name = file.split('/').pop()
  image_id = file_name.split('.')[0]
  idx = id2type[id2type['image_id']==image_id].index[0]
  t = id2type.iloc[idx]['dx']
  shutil.copy(src=file, dst=os.path.join(dataset_dir, t, file_name))

# split data to train,valid,test
ids = os.listdir(dataset_dir)
for idd in tqdm.notebook.tqdm(ids):
  if not os.path.isdir(os.path.join(dir, 'train', idd)):
    os.makedirs(os.path.join(dir, 'train', idd))
  if not os.path.isdir(os.path.join(dir, 'valid', idd)):
    os.makedirs(os.path.join(dir, 'valid', idd))
  if not os.path.isdir(os.path.join(dir, 'test', idd)):
    os.makedirs(os.path.join(dir, 'test', idd))

  add = os.path.join(dataset_dir, idd)
  files = os.listdir(add)
  train_val_files, test_files = train_test_split(files, test_size=.2, shuffle=True)
  train_files, val_files = train_test_split(train_val_files, test_size=.15, shuffle=True)

  print(f':::Organizing {idd} with {len(files)} files:::')
  print(f'Preparing trainig files for {idd}:\n')
  for file in tqdm.notebook.tqdm(train_files):
    src = os.path.join(add, file)
    dst = os.path.join(dir,'train', idd, file)
    shutil.copy(src, dst)
  print(f'Preparing validation files for {idd}:\n')
  for file in tqdm.notebook.tqdm(val_files):
    src = os.path.join(add, file)
    dst = os.path.join(dir,'valid', idd, file)
    shutil.copy(src, dst)
  print(f'Preparing test files for {idd}:\n')
  for file in tqdm.notebook.tqdm(test_files):
    src = os.path.join(add, file)
    dst = os.path.join(dir,'test', idd, file)
    shutil.copy(src, dst)
  print(f'=========================END of {idd} files=========================\n')

## Dataloader setup
train_dir = os.path.join(dir, 'train')
valid_dir = os.path.join(dir, 'valid')
test_dir = os.path.join(dir, 'test')

transform = transforms.Compose([transforms.Resize((224,224)), transforms.RandomRotation(10),
                                transforms.RandomVerticalFlip(.3),
                                transforms.RandomHorizontalFlip(.3),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                     std = [0.229, 0.224, 0.225])])
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

### Model definition
detector = models.resnext101_32x8d(pretrained=True)
for m in detector.children():
  # print(m.__class__.__name__)
  for p in m.parameters():
    p.requires_grad = False

for p in detector.fc.parameters():
  p.requires_grad = True
  print(p)

classifier = nn.Sequential(nn.Linear(2048, 1000), nn.ReLU(), nn.Dropout(.3),
                           nn.Linear(1000, 500),  nn.ReLU(), nn.Dropout(.3),
                           nn.Linear(500, 50), nn.ReLU(), nn.Dropout(.3),
                           nn.Linear(50, 7))
detector.fc = classifier
