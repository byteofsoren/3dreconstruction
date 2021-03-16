# Prevents circular imports.
from __future__ import annotations

import numpy as np
import cv2, yaml
import logging
import copy
import pandas as pd
from bcolor import bcolors
from typing import TYPE_CHECKING
from tabulate import tabulate
from pathlib import Path
from cv2 import aruco
from simple_term_menu import TerminalMenu  # Used to select display options
from shapely.geometry import Polygon # Calculations of the aria
from typing import Type # Used to be able to pass var:Type[object] to a function

# Try to prevent circular imports
if TYPE_CHECKING:
        from .Viewmod  import View
        from .Cornermod import Corner

# == Logging basic setup ===
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/atlasCL_Transfer.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===



class Linkable():
    """docstring for linkable"""
    name:str = "None given"
    """ Name is used mostly for debugging puprose """
    _ttype = None
    """ Tels what type the inhereted cass is """
    transfers:list
    """ Stores the transfer from a view to each corner in the view (id,Transfer) """
    Corner = None
    """ Stores the corner type to be accesed later """
    View   = None
    """ Stores the view type to be accesed later """
    temp_transfer:Transfer = None
    """ Used to pass information over to _check_transfer() funciton """

    def __init__(self, name:str)->None:
        log.info(f"Link created {name}")
        from .Viewmod  import View
        from .Cornermod import Corner
        self.Corner = Corner
        self.View = View
        self.transfers = list()

        self.name = name
        if type(self) is Corner:
            self._ttype = Corner
        elif type(self) is View:
            self._ttype = View


    def check_transfer(self)->bool:
        """ This member function needs to be over
        written by teh inherited class else it rise an Exception error
        while using the add_transfer() function call.

        :raises Exception: Need to implement by the child.
        """
        raise Exception("check_transfer() need to be over written")


    def add_transfer(self, tf:Type[Transfer]):
        """ Adds a transfer to the inhereted class

        :param tf: In the transfer needed to add to to the class
            Observe this function CALS

                check_transfer():

            witc must be overwritten
            by the implemented class
        """
        self.temp_transfer = tf
        if self.check_transfer(): # <- check_transfer is over written
            log.info(f"Transfer {bcolors.OK}[OK]{bcolors.END}")
            self.transfers.append(tf)
        # else:
        #     log.info(f"Transfer {bcolors.ERR}[FAIL]{bcolors.END}")
        #     raise ValueError(f"Transfer {str(tf)} was not axepted")

    def len_transfers(self)->int:
        return len(self.transfers)






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
    name:str
    """ Name is used for debugging purpose """
    source:None
    """Link form one target"""
    target:None
    """Link to the next target"""
    rvec:np.ndarray
    """Rotation vector"""
    tvec:np.ndarray
    """Translation vector"""
    wheight:float = 0
    """Optional How reliable camera to corner transfer this connection is"""
    tf_length:int = 10e10
    """ Transfer length to origin (coner 10) -> (corner 3) -> (corner 0) => tf_length=2 """

    def __init__(self, source:Type[Linkable], target:Type[Linkable], tvec=None, rvec=None):
        self.name = f"Transfer source:{source.name}, target:{target.name}"
        log.info(f"Init Transfer {self.name}")
        self.source = source
        self.target = target
        log.info(f"---tvec.shape={tvec.shape}")
        log.info(f"---rvec.shape={rvec.shape}")
        self.tvec=np.array(tvec[0])
        self.rvec=np.array(rvec[0])
        log.info(f"tvec={self.tvec} shape {self.tvec.shape}")
        log.info(f"rvec={self.rvec} shape {self.rvec.shape}")
        # convert rot vector to rot matrix both do: markerworld -> cam-world
        R, jacobian = cv2.Rodrigues(rvec)
        log.info(f"R={R} shape={R.shape}")
        # breakpoint()
        self.T = np.vstack([np.hstack([R,tvec[0].T]),[0,0,0,1]])
        log.info(f"__init__ end self.T =\n{self.T}")
        # https://answers.ros.org/question/314828/opencv-camera-rvec-tvec-to-ros-world-pose/

    def __str__(self):
        return self.name

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

    def _inversePerspective(self):
        R, _ = cv2.Rodrigues(self.rvec.reshape((3,1)))
        R = np.matrix(R).T
        invTvec = np.dot(-R, np.matrix(self.tvec.reshape((3,1))))
        invRvec = cv2.Rodrigues(R)
        return invTvec, invRvec

    def __sub__(self, other):
        """ To solve the marker to marker calulations for the tvec and
            rvec the vector calulation is for marker A and B with camrea C
            AB = AC - BC
            Thus by ower writing the sub part of this object the this calculation
            in python is just Transfer1 - Transfer2
            :param other: Is an other Transfer object
        """
        if type(other) is Transfer:
            log.info("__sub__")
            invTvec, invRvec = other._inversePerspective()
            orgTvec, orgRvec = self._inversePerspective()
            rvec = self.rvec.reshape((3,1))
            tvec = self.tvec.reshape((3,1))
            info = cv2.composeRT(
                    rvec,
                    tvec,
                    invRvec[0],
                    invTvec)
            compRvec:np.ndarray = info[0]
            compTvec:np.ndarray = info[1]
            compRvec = compRvec.reshape((1,1,3))
            compTvec = compTvec.reshape((1,1,3))
            log.info(f"compRvec=\n{compRvec}\ncompTvec=\n{compTvec}")
            # breakpoint()
            log.info("--Create a new transfer--")
            tf = Transfer(self.source,other.source,compTvec,compRvec)
            log.info(f"__sub__ return {tf}")
            return tf

    @property
    def Tinv(self):
        """Invested Transfer matrix for this link"""
        return np.linalg.inv(self.T)
