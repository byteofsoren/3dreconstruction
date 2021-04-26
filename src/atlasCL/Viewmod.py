from cv2 import aruco
import numpy as np
import cv2, yaml
import logging
import copy
import pandas as pd
from bcolor import bcolors
from tabulate import tabulate
from pathlib import Path
# import aruco
from simple_term_menu import TerminalMenu  # Used to select display options
from shapely.geometry import Polygon # Calculations of the aria
from typing import Type # Used to be able to pass var:Type[object] to a function

# Local imports
# from ..camera import camera

from   .Cornermod import Corner
from .Transfermod import Transfer
from .Transfermod import Linkable
# Atlas dependent classes import


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
class View(Linkable):
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

    # _TRvecs:dict = dict()
    # """Privae: A dict with IDS as key and rvec,tvec,corners as values"""
    fname:str=None
    """ Filename for this view  """

    ids:None
    """IDS is the list of ids in the view"""
    _corners2d:np.ndarray
    """Private: All 2D points for each corner in the view"""
    origin_aruco:int = 0
    """Private"""
    back_trace:list
    #_corners = None # deprecated
    # """ Stored result from aruco.detectMarkers() function.
    #     Not to be confused with self.corners"""
    # Transfers = dict()
    # """A dict of (tranfer())[ids] that stores the Transfer from a view to its markers"""


    def __init__(self, name:str, img:np.ndarray, arucodict, arucoparam, corner_size,camera):
        super(View, self).__init__(name=f"View {str(name)}")
        log.info(f"-- View with name:{name} created --")
        self.img = img
        self.back_trace = list()
        self.TRvecs = dict()
        self.fname = name
        self.id = name
        self.corner_size = corner_size
        self.camera = camera
        # read the config file.
        with open('./atlas.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        # settings needed
        self._conf = conf['view_obj']
        self.origin_aruco = conf['origin_aruco']
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
        # self._ids, self._corners2d = zip(*(sorted(ziped)))
        self._ids, self._corners2d = zip(*ziped)
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
            log.error(f"[FAIL] {self.name} did not pass ids was {self._ids}")
            raise ValueError("CornerERROR")

        log.info(f"[OK] {self.name} passed CornerERROR")
        # breakpoint()
        for i in range(0,len(self._ids)):
            id = self.ids[i]
            rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(
            corners=corners[i], #Corners used
            markerLength=self.corner_size,
            cameraMatrix=mtx,
            distCoeffs=dist)
            # Store connections in dict
            try:
                self.set_TRvec(id,tvec,rvec,self._corners2d[i])
            except Exception as e:
                    log.error("----------- FAIL ------------")
                    raise e
        aruco.drawDetectedMarkers(self.img,self._corners2d,borderColor=(0,255,0))


    def __str__(self):
        return f"View filename={self.name}  aruco ids = {self.ids}"

    # def generate_transfer(self, origin=0):
    #     """generate tarnsfer function for this view."""
    #     # direction of transfers:
    #     # Origin -> corner -> corner -> view
    #     # Diretion of discovery:
    #     # view -> corner -> corner -> origin
    #     # The problem is that the back trace may not mach the discovery:
    #     # O -> C2 -> c3 -> view
    #     #   \           /
    #     #     <- c5 <-
    #     tf:Transfer
    #     # breakpoint()
    #     log.info(f"{bcolors.OKBLUE}-- {self.name} --{bcolors.END}")
    #     low = None
    #     if len(self.transfers) == 0 or self.transfers is None:
    #         log.warn(f"{bcolors.INF}{self.name}{bcolors.END} Has no transfers")
    #         return
    #     for tf in self.transfers:
    #         log.info(f"{bcolors.OK}Indexing aruco value for tf {tf.source.aruco_value}{bcolors.END}")
    #         if low is None:
    #             low = tf.source
    #         elif isinstance(tf.source, self.Corner) and tf.source.aruco_value < low.aruco_value:
    #             log.info(f"{bcolors.INF}{low.name}:{low.aruco_value}{bcolors.END} < {bcolors.OK}{tf.source.name}:{tf.source.aruco_value}{bcolors.END}")
    #             low = tf.source
    #     # if low is None:
    #     #     breakpoint()
    #     log.info(f"{bcolors.OKBLUE}{self.name}{bcolors.END} aruco_value= {low.aruco_value}")
    #     stack = list()
    #     stack.append(low)
    #     current = 10e3
    #     if not stack[-1].aruco_value == origin:
    #         while not current == origin:
    #             lc:Corner = stack[-1]
    #             low = None
    #             low_tf = None
    #             # find lowest aruco_value in targets
    #             if len(lc.transfers) == 0:
    #                 log.error(f"{bcolors.OKCYAN}{lc.name}{bcolors.ERR} Number of transfers to low {bcolors.END}")
    #                 break
    #             else:
    #                 log.error(f"{lc.name} OK")
    #             log.info(f"{self.name} -> transfers len {len(lc.transfers)} {bcolors.SIGN.OK}")
    #             for tf in lc.transfers:
    #                 if low is None:
    #                     low = tf.target
    #                     low_tf = tf
    #                 elif isinstance(tf.target, self.Corner) and tf.target.aruco_value < low.aruco_value:
    #                     low = tf.target
    #                     low_tf = tf
    #             # Now low contains an target Corner with the lowest aruco_value:
    #             # Verify that there is a transfer:
    #             # ----              ----
    #             # |  | <----------  |  |
    #             # |  | ---------->  |  |
    #             # ----              ----
    #             # But as the low target has several connections it self
    #             # the back connection must be searched for.
    #             found = False
    #             for tf in low.transfers:
    #                 if tf.target == lc:
    #                     found = True
    #                     log.info(f"tf.target = lc {bcolors.OK}[OK]{bcolors.END}")
    #                     break
    #             if not found:
    #                 try:
    #                     # This do not work because low is a Corner rather then a transfer.
    #                     lc.transfers.pop(lc.transfers.index(low_tf))
    #                 except ValueError as e:
    #                     log.error(f"low {bcolors.ERR}{low.name}{bcolors.END} not in {[tf.name for tf in lc.transfers]}")
    #                     # breakpoint()
    #                     break
    #             else:
    #                 log.error(f"{bcolors.ERR}No low transfer match lc{bcolors.END}")
    #                 stack.append(low)
    #                 current = low.aruco_value
    #             # Now pruning is done
    #     # else:
    #     #     breakpoint()
    #     # breakpoint()
    #     for it in stack:
    #         print(f"{it.name} -> ")
    #         for tf in it.transfers:
    #             print(f"|- {tf.name}")



    def set_TRvec(self,id:int, tvec,rvec,corners2d):
        """ Sets TRvecs  containing tvec,rvec and corners for a aruco id
            :parma id: Aruco id
            :param tvec: Transfer vector shape (3,1)
            :param rvec: Rodrigues rotation vector shape (3,1)
            :parma corners2d: 4 points in 2D where the Aruco corner is in the image.
        """
        if isinstance(id, (int,np.integer,np.uint)) and id in self.ids:
            log.info(f"Added id:{id} to {self.name} {self.ids}")
            self.TRvecs[id] = (tvec,rvec,corners2d)
        else:
            log.warn("Tried to add id:{id} but it was not in the list of ids:{self.ids}")
            raise ValueError("Id:{id} is not in ids:{self.ids}")

    def get_TRvec(self, id:int):
        log.info(f"get_TRvec id={id} {self.ids} {self.TRvecs.keys()}")
        if id in self.ids and id in self.TRvecs.keys():
            return self.TRvecs[id]
        else:
            log.error(f"ID {id} not in TRvecs {self.TRvecs.keys()}")
            raise ValueError(f"The id {id} was not found in TRvecs {self.TRvecs.keys()}")


    def check_transfer(self)->bool:
        tf:Transfer = self.temp_transfer # Validate tf
        valid_target = tf.target is self
        valid_source = type(tf.source) is self.Corner and tf.source.id in self.ids
        log.info(f"Transfer from {str(tf.source)} to {str(tf.target)} is {'a valid' if valid_target and valid_source else 'Not a valid'} transfer")
        return valid_target and valid_source

# == View END ==

