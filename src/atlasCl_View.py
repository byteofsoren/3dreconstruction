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
# Dependency import for liknable
from atlasCL_Transfer import linkable

# == Logging basic setup ===
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/atlasCL_view.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===

# == View START ==
class View(linkable):
    """The view object contains the image and its connections to aruc corners
    The view object takes the following arguments

    :param str name: is teh name of the file tex img1.png
    :param np.ndarray img: Is the acual content of the file as an ndarray
    :param list arucodict: Is a link to the aruco corners used in this project.
    :param arucoparam: Aruco parametsers
    :param float corner_size: is the physical sice of the corner in meter
    :param camera camera: I connection to the camera object.
    :raises CornerERROR: If there are no corners in the view.
    """

    corners:dict = dict()
    """A dict with IDS as key and rvec,tvec,corners as values"""
    ids:None
    """IDS is the list of ids in the view"""
    _corners2d:np.ndarray
    """All 2D points for each corner in the view"""
    _origin_aruco:int = 0
    """  """
    #_corners = None # deprecated
    # """ Stored result from aruco.detectMarkers() function.
    #     Not to be confused with self.corners"""
    # Transfers = dict()
    # """A dict of (tranfer())[ids] that stores the Transfer from a view to its markers"""


    def __init__(self, name:str, img:np.ndarray, arucodict, arucoparam, corner_size,camera):
        super(view, self).__init__(name=f"View {str(name)}")
        log.info(f"-- View with name:{name} created --")
        self.img = img
        self.corner_size = corner_size
        self.camera = camera
        # read the config file.
        with open('./atlas.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        # settings needed
        self._conf = conf['view_obj']
        self._origin_aruco = conf['origin_aruco']
        self._gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        self._aruco = arucodict
        self._parameters = arucoparam
        mtx = self.camera._camera_matrix
        dist = self.camera._distortion_coefficients0
        if conf['view_obj']['use_camera_matrix']:
            tmp = aruco.detectMarkers(
                    self._gray,
                    self._aruco,
                    parameters=self._parameters,
                    cameraMatrix=mtx, #Using camera matrix produces nan
                    distCoeff=dist   #Do not use these here.
                    )
        else:
            tmp = aruco.detectMarkers(
                    self._gray,
                    self._aruco,
                    parameters=self._parameters,
                    )
        corners, self._ids, self._rejectedImgPoints = tmp #corner def
        # Sort the corners.
        ziped=zip(self._ids, corners)
        self._ids, self._corners2d = zip(*(sorted(ziped)))
        self._axis_matrix = np.float32(
                [[-0.01, -0.01, 0],
                [-0.01, 0.01, 0],
                [0.01, -0.01, 0],
                [0.01, 0.01, 0]])

        log.info(f"ids: {self._ids}")
        log.info(f"#corners: {corners}")
        if not self._ids is None:
            """ Flatten the list incase of problem """
            self.ids = [x[0] for x in self._ids]
        else:
            raise Exception("CornerERROR")

        log.info("Corners in {self.name}")
        for i in range(0,len(self._ids)):
            rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(
                    corners=corners[i], #Corners used
                    markerLength=self.corner_size,
                    cameraMatrix=mtx,
                    distCoeffs=dist)
            # Store connections in dict
            self.corners[self.ids[i]] = (tvec,rvec,self._corners2d[i],'doc: tvec,rvec,corner2d')
            aruco.drawDetectedMarkers(self.img,self._corners2d,borderColor=(0,255,0))


    def get_Transfer(self,id:int):
        """Returns the transfer matrix in this view"""
        if id in self.transfers:
            return self.transfers[id]
        else:
            raise ValueError(f'id={id} has no transfer in this object')


    def __str__(self):
        return f"View filename={self.name}  aruco ids = {self.ids}"


# == View END ==

