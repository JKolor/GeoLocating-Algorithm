"""
This file contains the code for the Google Colab notebook which connects to the Anvil App via Uplink.
The Google Colab notebook performs the predictions for both windows in the app.
The file contains the following sections of code:
-> Importing Libraries and Aetting Up Drivers
-> Loading 2 Deep Learning models: the 1 image integration model and the 4 image integration model.
-> Defining Necessary Variables and functions.
-> Connecting to the Anvil App's via server Uplink.
-> Defining the functions which will be used during the predictions.
"""

# Install selenium to the Google Colab Notebook. Selinium helps with automating browser and setting up virtual webdrivers.
!pip install selenium

# Import necessary Libraries
# Import libraries which will be used to activate and work with the webdriver.
from selenium import webdriver
import time
import sys

# Import libraries used for file handling.
import os
import re
import PIL

# Import tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import libraries used for plotting graphs along with numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# Install chromium for the webdriver. Chromium is a an executable that the selinium webdriver uses to control Google Chrome.
# Since the google colab servers run on Linux we need to Install the chromium web browser package using shelll commands from the Debian Buster repository.
%%shell
# Add debian buster
cat > /etc/apt/sources.list.d/debian.list <<'EOF'
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster.gpg] http://deb.debian.org/debian buster main
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster-updates.gpg] http://deb.debian.org/debian buster-updates main
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-security-buster.gpg] http://deb.debian.org/debian-security buster/updates main
EOF

# Add keys which are used to verify the authenticity of the packages that will be installed from the repository.
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys DCC9EFBF77E11517
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 112695A0E562B32A

apt-key export 77E11517 | gpg --dearmour -o /usr/share/keyrings/debian-buster.gpg
apt-key export 22F3D138 | gpg --dearmour -o /usr/share/keyrings/debian-buster-updates.gpg
apt-key export E562B32A | gpg --dearmour -o /usr/share/keyrings/debian-security-buster.gpg

# Prefer debian repo for chromium* packages only
# Note the double-blank lines between entries
cat > /etc/apt/preferences.d/chromium.pref << 'EOF'
Package: *
Pin: release a=eoan
Pin-Priority: 500

Package: *
Pin: origin "deb.debian.org"
Pin-Priority: 300

Package: chromium*
Pin: origin "deb.debian.org"
Pin-Priority: 700
EOF

# Install chromium and chromium-driver using the following commands
!apt-get update
!apt-get install chromium chromium-driver

# Set up the Selinium Chrome webdriver
def web_driver():
    options = webdriver.ChromeOptions() # Specify various Chrome browser options
    options.add_argument("--verbose") # enable verbose logging
    options.add_argument('--no-sandbox')  # disable the Chrome sandbox
    options.add_argument('--headless')  # run chrome without a graphical user interface
    options.add_argument("--window-size=1920, 1200")  # Set the window size to 1920 pixels by 1200 pixels
    driver = webdriver.Chrome(options=options)  # Create the webdriver with the options above.
    return driver

# Connect to your google drive in order to download and load the 2 Deep Learning models.
from google.colab import drive
drive.mount('/content/drive')

# Load the 4 image integration model from your google drive using its specified path
model_4_images = tf.keras.models.load_model("/content/drive/MyDrive/Geo-Locating Project/CurrentModelMid")
model_4_images.summary()

# Load the 1 image integration modelfrom your google drive using its specified path
model_1_image = tf.keras.models.load_model("/content/drive/MyDrive/Geo-Locating Project/CurrentModel")
model_1_image.summary()

# Define a list of all of the 50 US state names in alphabetical order.
states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
          "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho",
          "Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana",
          "Maine","Maryland","Massachusetts","Michigan","Minnesota",
          "Mississippi","Missouri","Montana","Nebraska","Nevada",
          "New Hampshire","New Jersey","New Mexico","New York",
          "North Carolina","North Dakota","Ohio","Oklahoma","Oregon",
          "Pennsylvania","Rhode Island","South Carolina","South Dakota",
          "Tennessee","Texas","Utah","Vermont","Virginia","Washington",
          "West Virginia","Wisconsin","Wyoming"]

# Define a list of the 50 US state 2 character names in alphabetical order.
states_short = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

# Import from selinium the By class which helps locate elements with in a web document.
from selenium.webdriver.common.by import By

# The function "screenshot" gets the driver and an index which is the rotation degree of the image as parameters.
# The function later uses the driver to take a screenshot of the geoguessr canvas and save it as a png image in the notebook directory.
def screenshot(driver, index):
  if (os.path.isfile(f'canvas_{index}.png')):
    os.remove(f'canvas_{index}.png')
  with open(f'canvas_{index}.png', 'xb') as f:
    canvas = driver.find_element(By.TAG_NAME, 'canvas')
    f.write(canvas.screenshot_as_png)

# The function "rotate_user" gets the driver as a parameter and drags the courser a specified distance 5 times until the user canvas is rotated by 90 degrees.
def rotate_user(driver):
  main = driver.find_element(By.TAG_NAME,"main")
  for x in range(0, 5):
    action = webdriver.common.action_chains.ActionChains(driver)
    action = action.move_to_element(main)
    action = action.click_and_hold(main)
    action = action.move_by_offset(178, 0)
    action = action.release(main)
    action.perform()

# The function "parse_image" gets an image filename as a parameter, and returns a preprocessed image as a NumPy array of unsigned integers.
def parse_image(filename):
    image = keras.utils.load_img(filename)  # Load the image in a PIL format.
    image = image.crop((600,100,1500,1000)) # Resize the image to rid of GeoGuessr's graphical UI.
    image = image.resize((256, 256), resample = PIL.Image.BICUBIC)  # Reisze the image to match the image size needed for the models.
    input = keras.utils.img_to_array(image) # Turn PIL image to an array.
    input = np.expand_dims(input, axis = 0) # Add a batch dimention to the array.
    return input.astype(np.uint8)

# The function "prefix_to_filenames" gets a prefix as a parameter and returns the paths of all 4 images relating to that prefix.
def prefix_to_filenames(prefix):
    return (prefix + '_0.png', prefix + '_90.png', prefix + '_180.png',
            prefix + '_270.png')
  
# The function "grouped_streetview_dataset" gets list of files as an argument.
# The function creates a TensorFlow dataset consisting of 4 parsed images and an empty label so that the images could be predicted on by the model. 
def grouped_streetview_dataset(files):
    def parse_images(files):
        return (parse_image(files[0]), parse_image(files[1]),
                parse_image(files[2]), parse_image(files[3]))
    images = parse_images(files)
    d = tf.data.Dataset.from_tensor_slices(images)
    d = tf.data.Dataset.zip((d, tf.data.Dataset.from_tensor_slices(["Empty_Label"])))
    return d

# The function gets a TensorFlow dataset consisting of 4 images as an argument and prints all 4 next to each other.
def show_image(image):
  fig = plt.figure(figsize=(100, 25))
  for images, label in image:
    for i in range(4):
      plt.subplot(1, 4, i + 1)
      plt.imshow(images[i][0])
      plt.title(str(i*90) + "°", fontsize = 150)
      plt.axis("off")
  return fig

# Install Anvil Uplink in order for the Google Colab notebook to connect to the Anvil Application.
!pip install anvil-uplink

# Import from anvil the server module which will allow you to connect to the Anvil application.
import anvil.server
# Connect to the Anvil application server using a unique server uplink key.
anvil.server.connect("server_CUMPYW4BKIR6EGHUWCB6PUKT-VEPM2TD6F6IP5H4Z") 

# Install the kaledio package which provides useful functions for generating static images for web-based visualization libraries like Plotly.
!pip install kaleido

# Import more libaries and packages which will be used for image handling and to graph predictions.
import plotly.express as px
import plotly as plotly
from io import BytesIO
from PIL import Image
import anvil.media

"""
The function "classify_image" is a function which can be called externally by the anvil application and return variables to the app via the Anvil Server.
The function is used for the "ImageGuessr" window of the application, in which the user can upload an image to the app, for it to return the model's prediction.

The function gets a jpeg image as an argument, which is loaded and resized using the anvil, keras and PIL libraries.
The image is then turned into a numpy array and a batch dimension is added to it - so that its format would match the model's.

The image is predicted on by the 1 Image Integration Model.
The output of the model's prediction is a 50 cell list where each cell represents each of the model's prediction layer's nodes.

The list is then used to create a heat map of the US using the plotly library - The more red the state the more likely the photos were taken there.
The heat map is then converted to a jpeg BlobMedia object which will be used to display the map in the Anvil application.

The function returns the top 3 guesses which are the 3 states the image was most likely taken at, along with the heat map.
"""
@anvil.server.callable
def classify_image(file):
  # Load and process image
  with anvil.media.TempFile(file) as filename:
    image = keras.utils.load_img(filename)
  image = image.resize((256, 256), resample = PIL.Image.BICUBIC)
  
  # Turn Image to an array
  input = keras.utils.img_to_array(image)
  input = np.expand_dims(input, axis = 0)
  
  # Let the model Predict on the image
  prediction_array = model_1_image.predict(input)

  #draw the prediction map
  df = pd.DataFrame(states_short, columns = ['State'])
  df['Chances'] = prediction_array[0]
  fig = px.choropleth(df,
                    locations='State', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Chances',
                    color_continuous_scale="OrRd",
                    )
  pred_map = plotly.io.to_image(fig, format = "jpg")
  pred_map = anvil.BlobMedia(content_type="image/jpeg", content=pred_map)

  sorted_prediction_arrray = sorted(list(list(prediction_array)[0]), reverse = True )
  prediction_array = list(list(prediction_array)[0])
  # return the model's prediction.
  return f'1. {states[prediction_array.index(sorted_prediction_arrray[0])]}\n2. {states[prediction_array.index(sorted_prediction_arrray[1])]}\n3. {states[prediction_array.index(sorted_prediction_arrray[2])]}', pred_map

"""
The function "predict_geoguessr" is a function which can be called externally by the anvil application and return variables to the app via the Anvil Server.
The function is used for the "GeoGuessr" window of the application, in which the user pastes the URL of a geoguessr game and lets the app guess the user's location.

The function get the GeoGuessr game URL as an argument.
The function then activates a driver to open the GeoGuessr map via Google Chrome.
It then takes 4 screenshots of the Geoguesser Street View at 4 angles correspoing to the cardinal directions using the functios "screenshot" and "rotate_user".

The 4 images are then prepared for prediction by creating a datset using the "grouped_streetview_dataset" function and batching it.
The function later, uses the 4 Image Integration Model to predict the location of the image using the 4 screenshots.

The model's prediction is then turned into a heat map of the US , showing the likelihood of the user being in each state, by using the Plotly library.
The heat map is converted to a jpeg BlobMedia Object which will be used to display the image in the Anvil application.
The function returns the top three predicted states along with the heat map and a combined image of the four Street View images.
"""
@anvil.server.callable
def predict_geoguessr(GeoguessrMap):
  #Activate webdriver
  local_driver = web_driver()
  
  # Log into GeoGuessr
  local_driver.get(GeoguessrMap)
  time.sleep(2)
  
  # Take 4 screen shots of the game
  print(">> current status: ready to take screenshots")
  for index in range(0, 4):
    print(f">> current status: taking screenshot number {index + 1}")
    screenshot(local_driver, index*90)
    rotate_user(local_driver)
  
  print(">> current status: getting the images ready for prediction")
  prefix = "/content/canvas"
  files = prefix_to_filenames(prefix)

  # Turn the 4 images to a batch dataset
  sample = grouped_streetview_dataset(files)
  sample = sample.batch(1)

  # Use the show_image function to get a single picture showing the 4 images
  fig = show_image(sample)
  buf = BytesIO()
  fig.savefig(buf, dpi = 100)
  bytestring = buf.getvalue()
  picture = anvil.BlobMedia(content_type="image/jpeg", content=bytestring)
  
  # Let the model predict on the 4 images
  print(">> current status: predicting...")
  prediction_array = model_4_images.predict(sample)
  df = pd.DataFrame(states_short, columns = ['State'])
  df['Chances'] = prediction_array[0]

  # Use plotly express to output a heat map of the US - the more red the state the more likely the photo was taken there
  fig = px.choropleth(df,
                      locations='State', 
                      locationmode="USA-states", 
                      scope="usa",
                      color='Chances',
                      color_continuous_scale="OrRd",
                      )
  pred_map = plotly.io.to_image(fig, format = "jpg")
  pred_map = anvil.BlobMedia(content_type="image/jpeg", content=pred_map)

  sorted_prediction_arrray = sorted(list(list(prediction_array)[0]), reverse = True )
  prediction_array = list(list(prediction_array)[0])
  # return the model's prediction.
  return f'1. {states[prediction_array.index(sorted_prediction_arrray[0])]}\n2. {states[prediction_array.index(sorted_prediction_arrray[1])]}\n3. {states[prediction_array.index(sorted_prediction_arrray[2])]}', pred_map, picture
