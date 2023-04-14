from ._anvil_designer import GeoGuessrTemplate
from anvil import *
import anvil.server

class GeoGuessr(GeoGuessrTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)


  def predict_click(self, **event_args):
    """This method is called when the button is clicked"""
    result, pred_map, images_graph = anvil.server.call('predict_geoguessr', self.text_box_1.text)

    self.results_lable.foreground = '#333333'
    self.results_lable.text = result
    self.image_1.source = images_graph
    self.image_2.source = pred_map

  def text_box_1_pressed_enter(self, **event_args):
    """This method is called when the user presses Enter in this text box"""
    pass
