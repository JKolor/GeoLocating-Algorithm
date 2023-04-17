"""
This is code represents the class of the ImageGuessr Window of the Anvil Application.
In this window the user can upload an image that he has taken to the App and get its prediction about in which US state the image was taken.
The GeoGuessr class contains two interactive elements:
- a file loader via which the user will be able to upload the image
- a prediction button
The application class contains additional elements which are defined via the Anvil website and can be refrenced in order to add information to the website:
-> results_label is a label which will contain the App top three state predictions
-> image_1 is an element which will display the image uploaded by the user
-> image_2 is an element which will display the heat map returen by the external notebook post prediction
"""

# Import the designated anvil libraries and the ImageGuessrTemplate which is the design template for the specific app window.
from ._anvil_designer import ImageGuessrTemplate
from anvil import *
import anvil.server

# Define the ImageGuessr class which will be able to be refrenced and used in the Main class and gets the ImageGuessr Design Template as a parameter.
class ImageGuessr(ImageGuessrTemplate):
  
  # Take the properties from the ImageGuessr Design Template and implement them into the website with the __init__ constructor function.
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

  # The function "file_loader_1_change" is called whenever a file is uploaded to the fileloader and it gets the file as an argument.
  def file_loader_1_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    self.image_1.source = file  # Displays the image uploaded in the app using the "image_1" element.
    self.file_passed = True # Sets the "file_pass" variable to True to indicate that an image has been uploaded by the user.

  # The function "predict_click" is called whenever the "predict" button is clicked by the user.
  # The function checks whether an image has been uploaded and continues to call the external "classify_image" predicting function.
  # The "classify_image" function returns the prediction's results and a heat map of the US.
  # The "predict_click" then display's the returned variables on the app's UI.
  def predict_click(self, **event_args):
    # Perform a prediction if a file has been uploaded by the user.
    if (self.file_passed):
      result, pred_map = anvil.server.call('classify_image', self.image_1.source)

      self.results_lable.foreground = '#333333'
      self.results_lable.text = result  # Display's the prediction's result on the app
      self.image_2.source = pred_map  # Display's the heat map returned from the predicting function
    else:
      # If an image was not uploaded by the user, return ERROR
      self.results_lable.text = "Error"
