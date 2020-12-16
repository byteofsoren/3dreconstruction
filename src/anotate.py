import pandas as pd
import kivy
from readset import dataset
from atlas import view
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import StringProperty
from kivy.uix.dropdown import DropDown

# == Logging basic setup ===
import logging
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/anotate.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===



class anotateWidget(Widget):
    """docstring for anotateWidget"""
    def __init__(self, data):
        super(anotateWidget, self).__init__()
        self.data = data
        self.dropdown = DropDown()
        for row in self.data:
            log.info(f"Adding {row}")
            btn = Button(text=row, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: self.select_data(btn.text))
            self.dropdown.add_widget(btn)
        self.dropdown.bind(on_select=lambda instance, x: setattr(self.ids.select, 'text', x))

    def selectbtn(self):
        """Updates the selected set"""
        log.info("-selectbtn-")


    def select_data(self, db:str):
        log.info("-select_data-")
        self.dropdown.select(db)
        log.info(dir(self.ids))
        pass

    def quitbtn(self):
        """Closes the program"""
        log.info("-- EXIT --")
        App.get_running_app().stop()




class anotateApp(App):
    """The gui window for the app"""
    def build(self):
        log.info("run anotateApp.build")
        data = ['P1','P2','P3']
        return anotateWidget(data)



if __name__ == '__main__':
    anotateApp().run()
    # pass
