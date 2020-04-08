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

# following code is run only ONCE to sort out data based on lesion type
for file in tqdm.notebook.tqdm(glob.glob(orig_data_dir +'/*/*.jpg')):
  file_name = file.split('/').pop()
  image_id = file_name.split('.')[0]
  idx = id2type[id2type['image_id']==image_id].index[0]
  t = id2type.iloc[idx]['dx']
  shutil.copy(src=file, dst=os.path.join(dataset_dir, t, file_name))
