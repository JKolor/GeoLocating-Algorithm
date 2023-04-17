# GeoLocating-Algorithm
This repository contains the code, along with the code description for my attempt at creating 2 algorithms which are able to predict where an image was taken using Deep Learning.
The whole project consists of 2 deep learning models as well as a web app which uses both of them. The web App gives the user the ability to do two things:
- Upload an image and get the app's prediction regarding in which US state an image was taken (uses the "1-image-integration-model").
- Type of paste a link to a game of GeoGuessr and get the app's prediction regarding in which US state the user is in (uses the "4-image-integration-model").
There are two folders in this repository: "Application" and "GeoLocating-Models".
The "GeoLocating_Models" folder contains the implementaion of two Deep Learning Models:
- "1-image-integration-model" contains the code for the model which is a able to predict in which US state a single image has been taken.
- "4-image-integration-model" contains the code for the model which is able to predict in which US state, a 360 degree image split into 4 different images has been taken.
The "Application" folder contains the code implementaion of the Anvil Web App:
- "Anvil-Integration" contains the code of the Google Colab NoteBook responsible for taking in data from the app and performing predictions.
- The code for the 3 anvil app classes:
  - "Main" contains the code for the Main class of the Anvil App.
  - "GeoGuessr" contains the code for the GeoGuessr class of the Anvil App.
  - "ImageGuessr" contains the code for the ImageGuessr class of the Anvil App.
