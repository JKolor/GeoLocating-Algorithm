"""
This is code represents the class of the Main Window of the Anvil Application.
This "Main" window is able to host and display either the ImageGuessr or the GeoGuessr windows at one time.
It gives the user the ability to switch between the two windows using a sidebar with two links.
The Main class contains 2 interactive elements:
- a link which is used to switch to the ImageGuessr window
- a link which is used to switch to the GeoGuessr window
The application class contains additional elements which are defined via the Anvil website and can be refrenced in order to add information to the website:
-> label_2 is a label located in the application header which displays the user's current window
-> content_panel which is the "panel" which hosts either application windows. It does not include the main's header and side bars.
"""

# Import the designated anvil libraries as well as the Main Design Template and both the ImageGuessr and GeoGuessr classes.
from ._anvil_designer import MainTemplate
from anvil import *
import anvil.server
import anvil.tables as tables
from anvil.tables import app_tables
from ..ImageGuessr import ImageGuessr
from ..GeoGuessr import GeoGuessr

# Define a Main class which gets the Main Design Template as an argument.
class Main(MainTemplate):
  
  # Take the properties from the Main Design Template and implement them into the website with the __init__ constructor function.
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)
    self.label_2.text = "Image Guessr"  # Set the window label as "Image Guessr" since the first window that will be showed is the ImageGuessr window.
    self.label_2.visible = True
    self.content_panel.add_component(ImageGuessr()) # Display the ImageGuessr window in the app
    # Any code you write here will run before the form opens.
  
  # The function "ImageGuessr_link_click" is called whenever the user clicks on the "ImageGuessr" sidebar link.
  # The function clears the content_panel and updates it according to the ImageGuessr class imported before.
  def ImageGuessr_link_click(self, **event_args):
    """This method is called when the link is clicked"""
    self.content_panel.clear()
    self.label_2.text = "Image Guessr"
    self.content_panel.add_component(ImageGuessr())
    pass
  
  # The function "GeoGuessr_link_click" is called whenever the user clicks on "GeoGuessr" sidebar link.
  # The function clears the content_panel and updates it according to the GeoGuessr class imported before.
  def GeoGuessr_link_click(self, **event_args):
    """This method is called when the link is clicked"""
    self.content_panel.clear()
    self.label_2.text = "GeoGuessr"
    self.content_panel.add_component(GeoGuessr())
    pass
