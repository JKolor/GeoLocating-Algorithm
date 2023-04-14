from ._anvil_designer import MainTemplate
from anvil import *
import anvil.server
import anvil.tables as tables
from anvil.tables import app_tables
from ..ImageGuessr import ImageGuessr
from ..GeoGuessr import GeoGuessr

class Main(MainTemplate):

  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)
    self.label_2.text = "Image Guessr"
    self.label_2.visible = True
    self.content_panel.add_component(ImageGuessr())
    # Any code you write here will run before the form opens.

  def ImageGuessr_link_click(self, **event_args):
    """This method is called when the link is clicked"""
    self.content_panel.clear()
    self.label_2.text = "Image Guessr"
    self.content_panel.add_component(ImageGuessr())
    pass

  def GeoGuessr_link_click(self, **event_args):
    """This method is called when the link is clicked"""
    self.content_panel.clear()
    self.label_2.text = "GeoGuessr"
    self.content_panel.add_component(GeoGuessr())
    pass
