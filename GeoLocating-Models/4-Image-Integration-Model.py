"""
This python file contains the code for my 4 image classification deep learning model.
The code was imported from a google cloab notebook.
The file contains the following sections of code:
- Dataset Preparation
- Building The Siamese Model
- Training The Model On Dataset
- Performance Evaluation of The Model which includes graphing the results
"""

# Import necessary libraries for the model and dataset preparation
# Import tensorflow libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model as model

# Import libraries used for graphing along with numpy
import matplotlib.pyplot as plt
import numpy as np

# Import libraries used for working with the notebook's directory and for the dataset preparation
import os
import re
import random as random

# Both dataset, the 50States2K and the 50States10k should be located on your google drive in order for you to download them.
# The following 2 line peice of code imports google drive from google.colab and connects to your google drive library.
from google.colab import drive
drive.mount('/content/drive')

# After the notebook is connected to your drive the following code unzips both dataset into the notebook's server hardrive
!unzip "/content/drive/MyDrive/Geo-Locating Project/50States2K_test.zip" -d "/content/test_dataset" > /dev/null
!unzip "/content/drive/MyDrive/Geo-Locating Project/50States10K.zip" -d "/content/train_dataset" > /dev/null

# get_files: gets a directory to search for files in and a regex which is 
def get_files(cur_dir, regex):
    out_files = []
    for subdir,_,files in os.walk(cur_dir):
        out_files.extend([ os.path.join(subdir, file) for file in files if re.search(regex, file)])
    return out_files

def get_file_prefix(filename):
    return re.search('(.*/[a-z0-9A-Z_-]*)_[0-9]*.jpg', filename).group(1)

def all_directions_exist(prefix):
    return (os.path.isfile(prefix + '_0.jpg') and
            os.path.isfile(prefix + '_90.jpg') and
            os.path.isfile(prefix + '_180.jpg') and
            os.path.isfile(prefix + '_270.jpg'))

def delete_prefix(prefix, regex):
  if (os.path.exists(prefix + regex)):
    os.remove(prefix + regex)

def print_corrupted(dir, reget):
  for file in get_files(dir, reget):
    prefix = get_file_prefix(file)
    if (all_directions_exist(prefix) == False):
      print(prefix)
