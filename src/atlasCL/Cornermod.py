from cv2 import aruco
import numpy as np
import cv2, yaml
import logging
import copy
import pandas as pd
from bcolor import bcolors
#from tabulate import tabulate
#from pathlib import Path
# import aruco
# from camera import camera
from shapely.geometry import Polygon # Calculations of the aria
from typing import Type # Used to be able to pass var:Type[object] to a function

# # Import class dependany for likable object
from .Transfermod import Linkable
from .Transfermod import Transfer
# from .Transfermod import Linkable
# from Transfer import Transfer


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
class Corner(Linkable):
    """
    Class for aruco Corner relations

    :param int id: Is the id of the Corner
    :param atlas back_atlas: is a link back to the atlas object.
    """

    id:int = -1
    """ The aruco value for this corner """
    views:dict
    """Connection from the aruco to each view  where the aruco was observed. """
    corners:dict
    """Connection from corner to each corner Transfer type """
    aruco_value:int = 10e6 #
    """Connection value in reference to origin set large in the beginning"""


    def __init__(self, id:int, back_atlas):
        super(Corner, self).__init__(name=f"Corner {str(id)}")
        self.id:int = id
        self._connection:list = list()
        # self.name = f"Corner id:{id}"
        self.views = dict()
        self.corners = dict()
        self._back_atlas = back_atlas
        self._transfer_ids = list()
        log.info(f"Created corner {bcolors.OK}{id}{bcolors.END}")

    def check_transfer(self)->bool:
        tf:Transfer = self.temp_transfer # Validate tf
        valid_source = tf.source is self
        valid_target = type(tf.target) is self.Corner or type(tf.target) is self.View
        valid_member = True
        if type(tf.target) is self.Corner and not tf.target.id in self._transfer_ids:
            self._transfer_ids.append(tf.target.id)
        else:
            # breakpoint()
            valid_member = False
        log.info(f"""
                Transfer {str(tf.source.name)} -->
                {str(tf.target.name)} transfer
                {f'{bcolors.OK}[VALID] 'if valid_target and valid_source else f'{bcolors.ERR}[FAIL]'}{bcolors.END}
                {f'{bcolors.OK}[SOLO] 'if valid_member else f'{bcolors.ERR}[IN]'}{bcolors.END}
            """)

        if valid_target and valid_source and valid_member:
            return True
        else:
            return False
            # raise ValueError("Not a valid transfer")



# == Corner END ==
