import numpy as np
import pandas as pd
import torch
import os
import glob

# from google.colab import drive
# drive.mount('/content/gdrive')

dir = '/content/gdrive/My Drive/Colab Notebooks/Skin Cancer/'
orig_data_dir = dir + 'Original data'

## Preprocessing: sorting data based on lesion type
id2type = pd.read_csv(dir + 'HAM10000_metadata.csv')
types = id2type['dx'].unique().astype(str)

dataset_dir = dir + 'Dataset/'
if not os.path.isdir(dataset_dir):
  os.makedirs(dataset_dir)

for t in types:
  type_dir = dataset_dir + t
  if not os.path.isdir(type_dir):
    os.makedirs(type_dir)
