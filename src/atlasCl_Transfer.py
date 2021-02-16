
import numpy as np
import cv2, yaml
import logging
import copy
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from cv2 import aruco
from camera import camera
from simple_term_menu import TerminalMenu  # Used to select display options
from shapely.geometry import Polygon # Calculations of the aria
from typing import Type # Used to be able to pass var:Type[object] to a function

# == Logging basic setup ===
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/atlasCL_Transfer.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===



class linkable():
    """docstring for linkable"""
    name:str = "None given"
    def __init__(self, name):
        self.name = name


class Transfer():
    """
    Transfer object for transferring between different corners

    :param corner_link: link to a corner object
    :param tvec: translation parameters
    :param rvec: rotation parameter
    :param corner2d: [Optional] potentially used to weight a connection
    """
    T:np.ndarray # is the transfer matrix for this object
    """Transfer matrix for this link"""
    fromLink:None
    """Link form one target"""
    toLink:None
    """Link to the next target"""
    rvec:np.ndarray
    """Rotation vector"""
    tvec:np.ndarray
    """Translation vector"""
    wheight:float = 0
    """Optional How reliable camera to corner transfer this connection is"""

    def __init__(self, fromLink, toLink, tvec=None, rvec=None, corner2d=None):
        self.fromLink = fromLink
        self.toLink = toLink
        self._corner2d = corner2d # Optional
        log.info(f"Created transfer to {corner_link.id}")
        self.tvec=np.array(tvec[0])
        self.rvec=np.array(rvec[0])
        log.info(f"tvec={self.tvec} shape {self.tvec.shape}")
        log.info(f"rvec={self.rvec} shape {self.rvec.shape}")
        # convert rot vector to rot matrix both do: markerworld -> cam-world
        R, jacobian = cv2.Rodrigues(rvec)
        log.info(f"R={R} shape={R.shape}")
        self.T = np.vstack([np.hstack([R,tvec[0].T]),[0,0,0,1]])
        # https://answers.ros.org/question/314828/opencv-camera-rvec-tvec-to-ros-world-pose/

    def __str__(self):
        return f"Transfer to {str(self.corner_link.__str__())}"

    def _wheighting(self)-> float:
        """Calculate the strength of the transfer
        """
        if self._conf['weight_calculation_method'] == 'simple':
            dist = 0
            for k in self.tvec[0][0]:
                dist += k**2
            dist = np.sqrt(dist)
            log.info(f"Transfer: Distance to corner is {dist}")
            self.wheight = dist

    @property
    def Tinv(self):
        """Invested Transfer matrix for this link"""
        return np.linalg.inv(self.T)
