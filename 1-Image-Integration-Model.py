"""
This python file contains the code for my 1 image classification deep learning model.
The code was imported from a google cloab notebook.
The file contains the following sections of code:
- Dataset Preparation
- Building The Model
- Training The Model On Dataset
- Performance Evaluation of The Model which includes graphing the results
"""

# import necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os


# Both dataset, the 50States2K and the 50States10k should be located on your google drive in order for you to download them.
# The following 2 line peice of code imports google drive from google.colab and connects to your google drive library.
from google.colab import drive
drive.mount('/content/drive')

# After the notebook is connected to your drive the following code unzips both dataset into the notebook's server hardrive
!unzip "/content/drive/MyDrive/Geo-Locating Project/50States2K_test.zip" -d "/content/test_dataset" > /dev/null
!unzip "/content/drive/MyDrive/Geo-Locating Project/50States10K.zip" -d "/content/train_dataset" > /dev/null

# Define 2 variables
# One of which contains the location of the 50States10K in the notebook direcotry - the dataset contains 500,000 google streetview images.
# One of which contains the location of the 50States2K in the notebook directory - the dataset contains 100,000 google streetview images.
train_dir = "/content/train_dataset"
test_dir = "/content/test_dataset/test_data"

# Set a batch size for the datset set and an image size of 256 by 256 pixels
BATCH_SIZE = 32
IMG_SIZE = (256, 256)

# Seperate the 50States10K datset into two different shuffled datsets in a 1/10 split.
# The training datset will contain 90% of files and the evaluation datset will contains 10% of the files
ds_train, ds_val = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            validation_split=0.1,
                                                            subset="both",
                                                            seed=42)

# Initialize the shuffled testing dataset from the 50States2K dataset.
ds_test = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

# Define a variable which contains the 50 states' name in an alphabetical order. Will be used when printing the labels.
class_names = ds_train.class_names

# Print a 4x3 array of random picked images from the train dataset
plt.figure(figsize=(10, 10))
for images, labels in ds_train.take(1):
  for i in range(12):
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Check and print how many batches of 32 images there are on the training datset
print('Number of train batches: %d' % tf.data.experimental.cardinality(ds_train))

# Prefetch all datasets in order for the training and testing faze to be more efficient
AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
ds_test = ds_test.prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.prefetch(buffer_size=AUTOTUNE)
