from ._anvil_designer import ImageGuessrTemplate
from anvil import *
import anvil.server

class ImageGuessr(ImageGuessrTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run before the form opens.
  def file_loader_1_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    self.image_1.source = file
    self.file_passed = True

  
  def predict_click(self, **event_args):
    """This method is called when the button is clicked"""
    if (self.file_passed):
      result, pred_map = anvil.server.call('classify_image', self.image_1.source)

      self.results_lable.foreground = '#333333'
      self.results_lable.text = result
      self.image_2.source = pred_map
    else:
      self.results_lable.text = "Error"
