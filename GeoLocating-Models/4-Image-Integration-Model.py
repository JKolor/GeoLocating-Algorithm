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

"""
There are 2500 samples (10000 unique images) in each of the 50 states to provide a total of 125000 samples (0.5 million images).
Each sample contains 4 rectified, 256×256 Street View images taken from the sample location, facing the cardinal directions:
North, West, South, East
This dataset contains multiple outliers (some samples containe 3 images instead of the ususal 4).
The following code indentifies these outliers and deletes them from the disk.
"""

# Both dataset, the 50States2K and the 50States10k should be located on your google drive in order for you to download them.
# The following 2 line peice of code imports google drive from google.colab and connects to your google drive library.
from google.colab import drive
drive.mount('/content/drive')

# After the notebook is connected to your drive the following code unzips both dataset into the notebook's server hardrive
!unzip "/content/drive/MyDrive/Geo-Locating Project/50States2K_test.zip" -d "/content/test_dataset" > /dev/null
!unzip "/content/drive/MyDrive/Geo-Locating Project/50States10K.zip" -d "/content/train_dataset" > /dev/null

# The function "get_files" gets a directory to search for files and a regex (regular expression)
# The function searches for files that match the given regex in the specified directory and returns a list of all file paths that match.
def get_files(cur_dir, regex):
    out_files = []
    for subdir,_,files in os.walk(cur_dir):
        out_files.extend([ os.path.join(subdir, file) for file in files if re.search(regex, file)])
    return out_files

# The function "get_file_prefix" gets a filename as a parameter and returns a string representing the prefix of the file.
def get_file_prefix(filename):
    return re.search('(.*/[a-z0-9A-Z_-]*)_[0-9]*.jpg', filename).group(1)

# The function "all_directions_exist" gets a file prefix which represents a single sample, and checks if all 4 images belonging to the sample exist.
def all_directions_exist(prefix):
    return (os.path.isfile(prefix + '_0.jpg') and
            os.path.isfile(prefix + '_90.jpg') and
            os.path.isfile(prefix + '_180.jpg') and
            os.path.isfile(prefix + '_270.jpg'))

# The "delete_preix" function gets a prefix and a regex (regular expression) as parameters.
# The function checks if the "prefix+regex" exist in the current directory and if so deletes them from the directory.
def delete_prefix(prefix, regex):
  if (os.path.exists(prefix + regex)):
    os.remove(prefix + regex)

# The "print_corrupted" function gets a directory path and a regex, and then proceeds to iterate over all files in the directory matching the regex.
# The function then gets the prefix for the specific regex and if not all files relating to the prefix exist it prints out the prefix.
def print_corrupted(dir, regex):
  for file in get_files(dir, regex):
    prefix = get_file_prefix(file)
    if (all_directions_exist(prefix) == False):
      print(prefix)

# Set the directory path for both the training dataset and the testing dataset in two variables.
train_dir = "/content/train_dataset"
test_dir = "/content/test_dataset/test_data"

# Go over all files in both datasets and check whether there are any outliers using the function written above.
print("Train Corrupted Files:")
print_corrupted(train_dir, '_0.jpg')
print_corrupted(train_dir, '_90.jpg')
print_corrupted(train_dir, '_180.jpg')
print_corrupted(train_dir, '_270.jpg')

print("\nTest Corrupted Files:")
print_corrupted(test_dir, '_0.jpg')
print_corrupted(test_dir, '_90.jpg')
print_corrupted(test_dir, '_180.jpg')
print_corrupted(test_dir, '_270.jpg')

# The function "find_delete_corrupted_file" gets a directory path and a regex as parameters
# The function interates over all files in the directory that match the regex and gets the prefix for each one.
# The function deletes all files relating to the prefix (sample) if not all four images relating to that prefix exist.
def find_delete_corrupted_file(dir, regex):
  for file in get_files(dir, regex):
    prefix = get_file_prefix(file)
    if (all_directions_exist(prefix) == False):
      delete_prefix(prefix, '_0.jpg')
      delete_prefix(prefix, '_90.jpg')
      delete_prefix(prefix, '_180.jpg')
      delete_prefix(prefix, '_270.jpg')

# Delete all corrupted files in both dataset if there are any.
find_delete_corrupted_file(train_dir, '_0.jpg')
find_delete_corrupted_file(train_dir, '_90.jpg')
find_delete_corrupted_file(train_dir, '_180.jpg')
find_delete_corrupted_file(train_dir, '_270.jpg')

find_delete_corrupted_file(test_dir, '_0.jpg')
find_delete_corrupted_file(test_dir, '_90.jpg')
find_delete_corrupted_file(test_dir, '_180.jpg')
find_delete_corrupted_file(test_dir, '_270.jpg')

# Go over files in both datasets and make sure there are no outliers left.
print("Train Corrupted Files:")
print_corrupted(train_dir, '_0.jpg')
print_corrupted(train_dir, '_90.jpg')
print_corrupted(train_dir, '_180.jpg')
print_corrupted(train_dir, '_270.jpg')

print("\nTest Corrupted Files:")
print_corrupted(test_dir, '_0.jpg')
print_corrupted(test_dir, '_90.jpg')
print_corrupted(test_dir, '_180.jpg')
print_corrupted(test_dir, '_270.jpg')

# The function "parse_images" gets a filename as a parameter, reads the file contents and decodes the image to an image tensor.
# The function continues to set the shape of the image tensor to [256, 256, 3] and return it.
def parse_image(filename):
    filecontents = tf.io.read_file(filename)
    jpeg = tf.image.decode_jpeg(filecontents)
    jpeg.set_shape([256, 256, 3])
    return jpeg

# The function "prefix_to_filenames" gets a prefix as a parameter and returns the paths of all 4 images relating to that prefix.
def prefix_to_filenames(prefix):
    return (prefix + '_0.jpg', prefix + '_90.jpg', prefix + '_180.jpg',
            prefix + '_270.jpg')

# The function "read_grouped_filenames_and_labels" gets a root as a parameter, the root is a string representing the root directory containing the images.
# The function iterates through all subdirectories in each dataset (each subdirectory represents a state) and groups each image file according to their label.
# The function returns a list of all image filenames, a list of their corresponding labels, and all of the 50 states' names found in the directory.
def read_grouped_filenames_and_labels(root):
    labeled_filenames = []
    all_labels = []
    for dir in sorted(os.listdir(root)):
        cur_dir = os.path.join(root, dir)
        if os.path.isdir(cur_dir):
            label = dir
            all_labels.append(label)
            file_prefixes = [ get_file_prefix(file) for file in get_files(cur_dir, '_0.jpg') ]
            cur_files = [ prefix_to_filenames(prefix) for prefix in file_prefixes
                          if all_directions_exist(prefix) ]
            labeled_filenames.extend([ (filenames, len(all_labels)-1)
                                       for filenames in cur_files ])
    files = [ tmp[0] for tmp in labeled_filenames ]
    labels = [ tmp[1] for tmp in labeled_filenames ]
    print(labels)

    return files,labels, all_labels,

# The function "split_train_val" gets a list of filenames and a list of their corresponding labels and continues to split both lists by a given fraction. 
# The function will returns 4 shuffled lists: training filenames, training labels, validation filenames, and validation labels.
def split_train_val(files, labels):
    fraction = 0.1
    zipped = list(zip(files, labels))
    random.shuffle(zipped)
    validation_size = int(len(zipped)*fraction)
    val = zipped[:validation_size]
    train = zipped[validation_size:]
    train_files = [ element[0] for element in train ]
    train_labels = [ element[1] for element in train ]
    val_files = [ element[0] for element in val ]
    val_labels = [ element[1] for element in val ]

    return train_files, train_labels, val_files, val_labels

# The function "shuffle_ds" gets a list of image filenams and a list of their corresponding labels as parameters.
# The function shuffles both parameters as one and returns both shuffled lists back.
def shuffle_ds(files, labels):
  zipped = list(zip(files, labels))
  random.shuffle(zipped)
  files = [ element[0] for element in zipped ]
  labels = [ element[1] for element in zipped ]
  return files, labels

# The function "grouped_streetview_dataset" get a list of image filenames, and a list of their corresponding labels indexs as parameters.
# The function creates a TensorFlow dataset consisting of tuples of 4 parsed images and their corresponding label indexs. 
def grouped_streetview_dataset(files, labels):
    def parse_images(files):
        return (parse_image(files[0]), parse_image(files[1]),
                parse_image(files[2]), parse_image(files[3]))
    filename_dataset = tf.data.Dataset.from_tensor_slices(files)
    d = filename_dataset.map(parse_images)
    d = tf.data.Dataset.zip((d, tf.data.Dataset.from_tensor_slices(labels)))
    return d

# Using the functions above the both the 50States10K and 50States2k datasets are split into 5 lists:
# 2 containing the filenames, 2 containing their corresponding label indexs and a list of all 50 states in alphabetical order.
train_files, train_labels, label_names = read_grouped_filenames_and_labels(train_dir)
test_files, test_labels, label_names = read_grouped_filenames_and_labels(test_dir)
print(label_names)

# Split and shuffle the 2 lists relating to the training dataset into 4 lists: 2 for the training dataset and 2 for the validation dataset.
train_files, train_labels, val_files, val_labels = split_train_val(train_files, train_labels)
# Shuffle both lists that will make up the testing dataset.
test_files, test_labels = shuffle_ds(test_files, test_labels)

# Using the function "grouped_streetview_dataset", create 3 TensorFlow datasets consisting of tuples of 4 images and their corresponding label indexs.
# These 3 datasets will be used for: testing, training and validation.
ds_test = grouped_streetview_dataset(test_files, test_labels)
ds_train = grouped_streetview_dataset(train_files, train_labels)
ds_val = grouped_streetview_dataset(val_files, val_labels)

# Print the number of elements in the training dataset - should contain 112,500 samples of 4 images.
num_elements = tf.data.experimental.cardinality(ds_train).numpy()
print("number of elements: " + str(num_elements))

# Batch each one of the 3 dataset into 32 sample sized batches.
ds_train = ds_train.batch(32)
ds_val = ds_val.batch(32)
ds_test = ds_test.batch(32)

# Print the shape of the training dataset. Its shape should correspond with the 2 other datasets.
print(ds_train)

# Print the number of elements in the training dataset - should contain about 3516 batches of 32 samples of 4 images.
num_elements = tf.data.experimental.cardinality(ds_train).numpy()
print("number of batches: " + str(num_elements))

# Print a randomly selected sample containg 4 images from the testing dataset.
plt.figure(figsize=(10, 10))
for images, labels in ds_test.take(1):
  for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(images[i][5])
    plt.title(label_names[labels[5]] + " | " + str(i*90) + "°")
    plt.axis("off")
    
# Prefetch all datasets in order for the training and testing faze to be more efficient
AUTOTUNE = tf.data.AUTOTUNE # tunes the buffer size dynamically at runtime
ds_test = ds_test.prefetch(buffer_size=AUTOTUNE)
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.prefetch(buffer_size=AUTOTUNE)

# Make the resnet preprocessing layer which will be used to modify the dataset's images according to the resnet model's needs.
preprocess_input = tf.keras.applications.resnet.preprocess_input

# DownLoad the RESNET50 model which will be used as the convolutional part of the network.
IMG_SHAPE = (256,256) + (3,)  # Set desired image shape
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Define the Resnet50 model as untrainable so that the training faze wouldn't take place for a long duration.
base_model.trainable = False

# Create a function which return a concatenated version of the 4 outputs of the siamese convolutional network.
def concat(input1, input2, input3, input4):
  concatted = tf.keras.layers.Concatenate()((input1, input2, input3, input4))
  return concatted

# Create each of the 4 subnets' input layers
image1 = layers.Input(name="image1", shape=(256, 256, 3))
image2 = layers.Input(name="image2", shape=(256, 256, 3))
image3 = layers.Input(name="image3", shape=(256, 256, 3))
image4 = layers.Input(name="image4", shape=(256, 256, 3))

# Create each of the 4 subnets' ResNet50 networks which will have a preprocessing layer before them.
output_0 = subnet(preprocess_input(image1))
output_90 = subnet(preprocess_input(image2))
output_180 = subnet(preprocess_input(image3))
output_270 = subnet(preprocess_input(image4))

# Combine the outputs of each of the 4 subnets using the "concat" function written above.
concated_layer = concat(output_0, output_90, output_180, output_270)

# Define "global_average_layer" as the Global Average Pooling Layer which will make a vector out of the concatenated siamese network's outputs.
global_average_layer = layers.GlobalAveragePooling2D()(concated_layer)

# Define a dropout layer after the Global Average layer to deal with overfit.
dropout_layer = tf.keras.layers.Dropout(0.2)(global_average_layer)

# Make a 50 node prediction dense layer which will be used to predict and classify each image at the end of the network.
prediction_layer = layers.Dense(units=50)(dropout_layer)

# Finalize the siamese network.
siamese_network = model(
    inputs=(image1, image2, image3, image4), outputs=prediction_layer
)

# See the finalized architecture of the siamese deep neural network.
siamese_network.summary()

# Compile the model: set a base learning rate (0.0001), optimizer (Adam), loss function (Sparse Catagorical Cross Entropy), and metrics.
# The "from_logits" is set to "True" since the activation function of the prediction layer is linear and so it needs to be converted to softmax.
base_learning_rate = 0.0001
siamese_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Set a callback which saves the model weights to a specific path on the user's google drive.
checkpoint_path = "/content/drive/MyDrive/Geo-Locating Project/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

"""
Load the previously trained model's weights if you desire to do some testing or continue training from your last training faze.
os.listdir(checkpoint_dir)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load(latest)
"""

# Train the model and set a training dataset, amount of epochs, validation dataset, and callbacks.
initial_epochs = 7
history = siamese_network.fit(ds_train,
                    epochs=initial_epochs,
                    validation_data = ds_val,
                    callbacks = cp_callback)

# Save the complete model including its wieghts to a desired path in your google drive directory
siamese_network.save(filepath = '/content/drive/MyDrive/Geo-Locating Project/CurrentModelMid')

"""
Load the entire model (which was saved on your drive) including its architecture if you desire to do some testing instead of training.
siamese_network = tf.keras.models.load_model("/content/drive/MyDrive/Geo-Locating Project/CurrentModelMid")
"""

# After training, graph the difference in accuracy and loss between training and validation
# Define 4 variables which will contain the accuracy and loss history for the training and validation datasets.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# Set a canvas size which will contain both graphs.
plt.figure(figsize=(8, 8))
# Plot the accuracy graph which will comapre between the model's accuracy on the training and validation datasets.
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),0.6])
plt.title('Training and Validation Accuracy')
# Plot the loss graph which will compare between the model's Loss on the training and validation datasets.
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,4])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# Display both graphs.
plt.show()

# Calculate accuracy and loss on the training dataset using the model.evaluate method and then print out both.
evaluations = siamese_network.evaluate(ds_test)
print(f"Accuracy: {str(evaluations[1]*100)[:5]}%\nLoss: {str(evaluations[0])[:4]}")

# Choose an index between 0 and 31 which will be used to select a random sample of images later.
chosen_index = random.randint(0, 31)

# Get a batch from the test dataset and let the model perform a prediction on it.
image_batch, label_batch = ds_test.as_numpy_iterator().next()
output = siamese_network.predict(ds_test.take(1))

# Import pandas which is a library used for creating a dataframe.
import pandas as pd
# Create a pandas dataframe which include the 2 letter name of each state in an alphabetical order.
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
df = pd.DataFrame(states, columns = ['State'])  # Set the states list as the first column
df['Chances'] = output[chosen_index]    # Set the ouput values of the randomly chosen image as the second column

# Print the all rows of the dataframe to see if everything is optimal - works when putting the line below in a single code cell on google colab.
df

# Use plotly express to output a heat map of the US according to the dataframe - the more red the state the more likely the photos were taken there.
import plotly.express as px
fig = px.choropleth(df,
                    locations='State', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Chances',
                    color_continuous_scale="OrRd",
                    )
fig.show()

# Print the predicted set of pictures along with their actual location and predicted location.
index = df['Chances'].idxmax()  # define index as the index of the predicted state
label = label_batch[chosen_index]   # define label as the index of the actual state.
plt.figure(figsize=(10, 10))
for images, labels in ds_test.take(1):
  for i in range(4):
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(images[i][chosen_index])
    plt.title(label_names[labels[chosen_index]] + " | " + str(i*90) + "°")
    plt.axis("off")

print(f"Predicted: {label_names[index]} ||  Actual: {label_names[label]}")

# Draw a label graph where each states accuracy is given.
# Create a new dataframe which will consist of the state's name as well as the model's accuracy for each state.
accuracy_df = pd.DataFrame(states, columns = ['State'])

# Set 3 variables in order to calculate the accuracy of the model: count, accuracy list, amount list.
# Accuracy_list is a list 50 cells long. Each cells represents a different state. Contains the number of correct guesses for each state.
accuracy_list = [0] * 50
# Amount_list is a list 50 cells long. Each cells represents a different state. Contains the number of guesses for each state.
amount_list = [0] * 50
count = 0

# Let the model predict on the test dataset and turn the output to a list.
predictions = siamese_network.predict(ds_test)
predictions = list(predictions)
# The following nested loop, iterates over each label in each batch in the test dataset and checks if the model's prediction are correct.
# The loop updates both the accuracy_list and amount_list accordingly.
for image_batch, label_batch in ds_test:
  for label in label_batch:
    prediction_index = list(predictions[count]).index(max(predictions[count]))
    print(f"prediction: {prediction_index} -> actual: {label}")
    if (label == prediction_index):
      accuracy_list[label] += 1 # Update the correct guess list
    amount_list[label] += 1 # Update the guess list 
    print(count)
    count += 1

# Create an empty list with 50 cells which will contain the model's accuracy for each state.
precentage_list = [0] * 50
# Iterate of the accuracy_list and calculate the model's accuracy for each state.
for i in range(len(accuracy_list)):
  precentage_list[i] = (accuracy_list[i]/amount_list[i]) * 100
# Add the precentage list as a column to the accuracy dataframe.
accuracy_df['Accuracy'] = precentage_list
# Sort the dataframe so the graph will be presentable.
accuracy_df_sorted = accuracy_df.sort_values(by=['Accuracy'])

# Plot and print the actual label graph.
ax = accuracy_df_sorted.plot.bar(x='State', y='Accuracy %', rot=0, figsize=(20,8))

# Use plotly express to output a heat map of the US according to the model's accuracy - the more yellow the state the higher the model's accuracy on it.
fig = px.choropleth(accuracy_df,
                    locations='State', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Accuracy',
                    color_continuous_scale="Plasma",
                    )
fig.show()
