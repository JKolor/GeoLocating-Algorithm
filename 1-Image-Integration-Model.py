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
AUTOTUNE = tf.data.AUTOTUNE # tunes the buffer size dynamically at runtime
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
ds_test = ds_test.prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.prefetch(buffer_size=AUTOTUNE)

# Make the resnet preprocessing layer which will be used to modify the dataset's image according to the resnet model's needs.
preprocess_input = tf.keras.applications.resnet.preprocess_input

# DownLoad the RESNET50 model which will be used as the convolutional part of the network.
IMG_SHAPE = (256,256) + (3,)  # Set desired image shape
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Define the Resnet50 model as untrainable so that the training faze wouldn't take place for a long duration.
base_model.trainable = False

# See the shape of the batch of images after the go through the ResNet50
image_batch, label_batch = next(iter(ds_train))
feature_batch = base_model(image_batch) # ResNet50 output
print(feature_batch.shape)

# Print the base model's architecture - ResNet50's architecture
print(base_model.summary())

# Define "global_average_layer" as the Global Average Pooling Layer which will make a vector out of each convolutionized image
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# See what will be the shape of a batch of the images after it goes through the Global Average Layer
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Make a 50 node prediction dense layer which will be used to predict and classify each image at the end of the network
prediction_layer = tf.keras.layers.Dense(50)
# See what will be the shape of the output of the final model after it gets a batch of images as input
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# Assmble all parts of the model including: Input layer, Preprocessing Layer, Base Model, Global Average Layer, Dropout Layer, Output Layer
inputs = tf.keras.Input(shape=(256, 256, 3))  # Set an input layer with a desired image input shape
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.05)(x)  # Set a dropout layer after the GA layer to deal with overfit.
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs) # Finalize the model

# Compile the model: set a base learning rate (0.0001), optimizer (Adam), loss function (Sparse Catagorical Cross Entropy), and metrics.
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Set a callback which saves the model weights to a specific path on the user's google drive
checkpoint_path = "/content/drive/MyDrive/Geo-Locating Project/cp_single_image.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path) # Set the chekpoint's dir
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

"""
Load the previously trained model's weights if you desire to do some testing or continue training from your last training faze.
os.listdir(checkpoint_dir)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load(latest)

Or load the entire model (which was saved on your drive) including its architecture.
model = tf.keras.models.load_model("/content/drive/MyDrive/Geo-Locating Project/CurrentModel")
"""

# See the finalized architecture of the deep neural network
model.summary()

# train the model and set a training dataset, amount of epochs, validation dataset, callbacks.
initial_epochs = 8
history = model.fit(ds_train,
                    epochs=initial_epochs,
                    validation_data = ds_val,
                    callbacks=cp_callback)

# Save the complete model including its wieghts to a desired path in your google drive directory
model.save('/content/drive/MyDrive/Geo-Locating Project/CurrentModel')

# After training graph the difference in accuracy and loss between training and validation
# Define 4 variables which will contains the accuracy and loss history for the training and validation datasets.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# Set a canvas size which will contain both graphs
plt.figure(figsize=(8, 8))
# Plot the accuracy graph which will comapre between the model's accuracy on the training and validation datasets.
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),0.4])
plt.title('Training and Validation Accuracy')
# Plot the loss graph which will compare between the model's accuracy on the training and validation datasets.
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

# Calculate accuracy and loss on the training dataset using the model.evaluate method and then print both.
evaluations = model.evaluate(ds_test)
print(f"Accuracy: {str(evaluations[1]*100)[:5]}%\nLoss: {str(evaluations[0])[:4]}")

# Get a batch from the test dataset and let the model perform a prediction on it
image_batch, label_batch = ds_test.as_numpy_iterator().next()
output = model.predict_on_batch(image_batch)
print(output.shape)

# Import pandas which is a library used for creating a dataframe
import pandas as pd
# Create a pandas dataframe which include the 2 letter name of each state in an alphabetical order.
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
df = pd.DataFrame(states, columns = ['State'])  # Set the states list as the first column
df['Chances'] = output[0] # Set the ouput values of the first image in the batch as the second column

# Print the first five rows of the dataframe to see if everything is optimal.
df.head()

# Use plotly express to output a heat map of the US according to the dataframe - the more red the state the more likely the photo was taken there
import plotly.express as px
fig = px.choropleth(df,
                    locations='State', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Chances',
                    color_continuous_scale="OrRd",
                    )
fig.show()

# Show the predicted picture along with what state the model predicted and its actal state.
index = df['Chances'].idxmax()  # define index as the index of the predicted state
image, label = image_batch[0], label_batch[0] # define image as the predicted image and label as the index of the actual state.
plt.figure(figsize=(25, 25))
ax = plt.subplot(3, 3, 0 + 1)
plt.imshow(image.astype("uint8"))
plt.title(f"Predicted: {class_names[index]} ||  Actual: {class_names[label]}")
plt.axis("off")

# Draw a label graph where each states accuracy is given
# Create a new dataframe which will consist of the state's name as well as the model's accuracy for each state
accuracy_df = pd.DataFrame(states, columns = ['State'])
# Set the test dataset as none shuffled so that we will be able to itirate over it and the model's predictions on it.
ds_test = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                            shuffle=False,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

# Let the model predict on the test dataset and turn the output to a list
predictions = model.predict(ds_test)
predictions = list(predictions)

# Set 3 variables in order to calculate the accuracy of the model: count, accuracy list, amount list.
count = 0
accuracy_list = [0] * 50  # A list 50 cells long. Each cells represents a different state. Contains the number of correct guesses for each state.
amount_list = [0] * 50  # A list 50 cells long. Each cells represents a different state. Contains the number of guesses for each state.
# The following nested loop, iterates over each label in each batch in the test dataset and checks if the model's prediction are correct.
for image_batch, label_batch in ds_test:
  for label in label_batch:
    prediction_index = list(predictions[count]).index(max(predictions[count]))
    print(f"prediction: {prediction_index} -> actual: {label}")
    if (label == prediction_index):
      accuracy_list[label] += 1 # Update the correct guess list
    amount_list[label] += 1 # Update the guess list 
    count += 1

# Create a list with 50 cells which will contain the model's accuracy for each state.
precentage_list = [0] * 50
# Iterate of the accuracy_list and calculate the model's accuracy for each state.
for i in range(len(accuracy_list)):
  precentage_list[i] = (accuracy_list[i]/amount_list[i]) * 100
# Add the precentage list as a column to the accuracy dataframe.
accuracy_df['Accuracy'] = precentage_list
# Sort the dataframe so the graph will be presentable.
accuracy_df_sorted = accuracy_df.sort_values(by=['Accuracy'])

# Plot and print the actual label graph
ax = accuracy_df_sorted.plot.bar(x='State', y='Accuracy', rot=0, figsize=(20,8))

# Use plotly express to output a heat map of the US according to the model's accuracy - the more yellow the state the higher the model's accuracy on it.
fig = px.choropleth(accuracy_df,
                    locations='State', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Accuracy',
                    color_continuous_scale="Plasma",
                    )
fig.show()
