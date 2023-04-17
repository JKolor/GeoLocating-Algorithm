"""
This is code represents the class of the GeoGuessr Window of the Anvil Application.
In this window the user can type or paste a link of a GeoGuessr game and get the App's prediction of the user's current location.
The GeoGuessr class contains two interactive elements:
- a text box used for pasting the link
- a prediction button
The application class contains additional elements which are defined via the Anvil website and can be refrenced in order to add information to the website:
-> results_label is a label which will contain the App top three state predictions
-> image_1 is an element which will display the 4 images taken by the external notebook during prediction
-> image_2 is an element which will display the heat map returen by the external notebook post prediction
"""

# Import the designated anvil libraries and the GeoGuessrTemplate which is the design template for the specific App window.
from ._anvil_designer import GeoGuessrTemplate
from anvil import *
import anvil.server

# Create the GeoGuessr class which will be able to get refrenced in other classes and gets the design template as a parameter.
class GeoGuessr(GeoGuessrTemplate):
  
  # Take the properties from the GeoGuessr Design Template and implement them into the website with the __init__ constructor function.
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

  # The "predict_click" function is triggered when the "predict" button in the website is clicked.
  # The function takes the typed link from the text box and calls the external predicting function "predict_geoguessr" which performs its prediction on the game.
  # The external function return both the results of the prediction, the heat map, and a single picture showing the 4 images which the function will show on the app.
  def predict_click(self, **event_args):
    """This method is called when the button is clicked"""
    result, pred_map, images_graph = anvil.server.call('predict_geoguessr', self.text_box_1.text)

    self.results_lable.foreground = '#333333' # Responsible for "coloring" the prediction result's foreground
    self.results_lable.text = result  # Display's the prediction's result on the App
    self.image_1.source = images_graph  # Display's the picture containing the 4 image's the predicting function has taken during prediction.
    self.image_2.source = pred_map  # Display's the heat map returned from the predicting function.
  
  # The "text_box_1_pressed_enter" function would be used if would want to perform a task when "enter" is pressed while on the text box.
  # Since I don't have a use for this type of task, the function is empty with the "pass" method.
  def text_box_1_pressed_enter(self, **event_args):
    """This method is called when the user presses Enter in this text box"""
    pass
