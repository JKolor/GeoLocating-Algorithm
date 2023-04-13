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
from selenium import webdriver
import time
import sys
import os
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import PIL

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import plotly.express as px

#install chromium for the webdriver
%%shell

# Add debian buster
cat > /etc/apt/sources.list.d/debian.list <<'EOF'
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster.gpg] http://deb.debian.org/debian buster main
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster-updates.gpg] http://deb.debian.org/debian buster-updates main
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-security-buster.gpg] http://deb.debian.org/debian-security buster/updates main
EOF

# Add keys
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
