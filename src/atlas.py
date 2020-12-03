import numpy as np
import cv2, yaml
import logging
import pandas as pd
from cv2 import aruco

# == Logging basic setup ===
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/atlas.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===


class view():
    """The view object contains the image and its connections to aruc corners"""
    corners:list = list()
    ids:None
    _name:str # Often the filename of the frame/camera angle

    def __init__(self, name:str, img:np.ndarray, arucodict, arucoparam):
        self._name=name
        self.img = img
        # read the config file.
        with open('./atlas.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        # settings needed
        self._gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # self._aruco = aruco.Dictionary_get(aruco.DICT_6X6_250)
        # self._parameters = aruco.DetectorParameters_create()
        self._aruco = arucodict
        self._parameters = arucoparam

    def __str__(self):
        return f"View object filename={self._name} shape={self.img.shape}"

    def dectect_corners(self):
        tmp = aruco.detectMarkers(self._gray, self._aruco, parameters=self._parameters)
        self.corners, self.ids, self._rejectedImgPoints = tmp


class projection():
    """The map object calculats the relations between view and aruco corners"""
    _corner_proj = dict()
    _views = list()
    def __init__(self):
        with open('./atlas.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)

    def add_view(self, view):
        """ Add a view to the map """
        log.info("Atlas add view {view}")
        view.dectect_corners()
        self._views.append(view)
        if not view.ids is None:
            if len(view.ids) <= 2:
                log.warn(f"<== Low length of 'view.ids={view.ids}' ==>")
            for id in view.ids:
                log.info(f"Adding {view} to aruco id={id[0]}")
                if id[0] in self._corner_proj:
                    self._corner_proj[id[0]].append(view)
                else:
                    self._corner_proj[id[0]] = list()
                    self._corner_proj[id[0]].append(view)
        else:
            log.warn(f"View contained Nan {view._name}")

    def view_atlas(self):
        """ returns an table reprecentation for the atlas. """
        log.info("-- View_atlas show table --")
        keys_toprow = sorted(self._corner_proj.keys())
        corner = self._corner_proj
        view_leftcol = self._views
        for key in keys_toprow:
            if key in corner.keys():
                pass # key in frame ok
            else:
                pass # ken not in frame









