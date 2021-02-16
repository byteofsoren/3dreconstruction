
import numpy as np
import cv2, yaml
import logging
import copy
import pandas as pd
#from tabulate import tabulate
#from pathlib import Path
from cv2 import aruco
from camera import camera
from shapely.geometry import Polygon # Calculations of the aria
from typing import Type # Used to be able to pass var:Type[object] to a function
# Import class dependany for likable object
from atlasCL_Transfer import linkable

# == Logging basic setup ===
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/atlasCL_Corner.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===


# == Corner START ==
class Corner(linkable):
    """
    Class for aruco Corner relations

    :param int id: Is the id of the Corner
    :param atlas back_atlas: is a link back to the atlas object.
    """

    views:dict = dict()
    """Connection from the aruco to each view  where the aruco was observed. """
    aruco_value:int = 10e6 #
    """Connection value in reference to origin set large in the beginning"""

    connection = dict()
    """The connection from corner to each view"""


    def __init__(self, id:int, back_atlas):
        super(Transfer, self).__init__(name=f"Corner {id}")
        self.id:int = id
        self.name = f"Corner id:{id}"
        self._back_atlas = back_atlas
        log.info(f"Created aruco corner id={id} connection:{self.connection}")

    def add_View(self, connect_View):
        """ Adds a view to a corner
        :param view: a view object pointer
        :param id: the id of the view???
        """
        name = connect_View.name
        if not name in self.views.keys():
            self.views[name] = connect_View
        else:
            print(f"{name} is connected to corner")



    def connect_Corner(self, connection):
        """ Connect the corner to a other corner using a Transfer matrix """
        pass


    def getconnections(self, sort=False)->list:
        """
            Returns a list of ids in the corner, if sort is True, then the
            returned list is sorted alphabetically.

            :param bool sort: If set to True the return list is sorted.
            :return: Returns a list
            :rtype: list
        """
        if sort:
            return sorted(self.connection.keys())
        else:
            return self.connection.keys()

    def is_connected(self)-> bool:
        """ Is the corner connected to other corners? """
        return len(self.connection) > 0

# == Corner END ==
