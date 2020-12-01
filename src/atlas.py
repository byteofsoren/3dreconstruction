import numpy as np
import cv2, yaml
import logging
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

    def __init__(self, img:np.ndarray, arucodict, arucoparam):
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
        return f"View object of shape {self.img.shape}"


    def dectect_corners(self):
        tmp = aruco.detectMarkers(gray, self._aruco, parameters=self._parameters)
        self.corners, self.ids, self._rejectedImgPoints = tmp


class projection():
    """The map object calculats the relations between view and aruco corners"""
    views:list = list()
    def __init__(self):
        with open('./atlas.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)

    def add_view(self, view):
        """ Add a view to the map """
        log.info("Atlas add view {view}")
        self.views.append(view)

    def calucate_views(self):
        """ Clulates all corners of from all the views """
        for v in self.views:
            v.dectect_corners()

