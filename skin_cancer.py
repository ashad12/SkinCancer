import numpy as np
import pandas as pd
import torch
import os
import glob

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
