import numpy as np
import cv2, yaml
import logging
import pandas as pd
from tabulate import tabulate
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

class corner():
    """Class for aruco corner relations"""
    class transfer():
        """Transferobject for tranfereing between different corners"""
        _T:np.ndarray # is the transfer matrix for this object

        def __init__(self, name:str):
            pass
    _views:list # connection from the aruco to each view
                # where the aruco was observed.
    def __init__(self, id:int):
        self.id:int = id
        log.info(f"Created aruco corner id={id}")

    def add_view(self, view):
        self._views.append(view)

class view():
    """The view object contains the image and its connections to aruc corners"""
    corners:list = list()
    ids:None
    name:str # Often the filename of the frame/camera angle
    # isorigin:bool = False
    _origin_aruco:int = 0

    def __init__(self, name:str, img:np.ndarray, arucodict, arucoparam):
        self.name=str(name)
        self.img = img
        # read the config file.
        with open('./atlas.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        # settings needed
        self._origin_aruco = conf['origin_aruco']
        self._gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # self._aruco = aruco.Dictionary_get(aruco.DICT_6X6_250)
        # self._parameters = aruco.DetectorParameters_create()
        self._aruco = arucodict
        self._parameters = arucoparam
        tmp = aruco.detectMarkers(self._gray, self._aruco, parameters=self._parameters)
        self.corners, self._ids, self._rejectedImgPoints = tmp
        # Flatten the list incase of problem
        log.info(f"ids: {self._ids}")
        if not self._ids is None:
            self.ids = [x[0] for x in self._ids]
        else:
            raise Exception("No corner in images")

    def __str__(self):
        return f"View object filename={self.name} pressent aruco ids = {self.ids}"

    # def dectect_corners(self):

class atlas():
    """The map object calculats the relations between view and aruco corners"""
    _corner_proj = dict()
    # _views = list()

    _views = dict()
    aruco_ids:list=list()  # Contains a list of aruco corner ids
                    # indentified in the set
    aruco_corners:list = list() # Stores the aruco corners in the atlas
    _aruco_origin_id:int = 0 # origin id set in atlas.yml
    _aruco_origin:corner # A handle to the origin aruco
    _confusion_atlas=False
    _confusion_frame:pd.DataFrame # Stores the confusion frame

    def __init__(self):
        with open('./atlas.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        self._aruco_origin_id = conf['origin_aruco']


    def add_view(self, view):
        """ Add a view to the map """
        log.info(f"#Atlas add view {view}")
        self._views[view.name] = view
        if not view.ids is None:
            for index in view.ids:
                if not index in self.aruco_ids:
                    log.info(f"#add {index}")
                    self.aruco_ids.append(index)


        self.aruco_ids = sorted(self.aruco_ids)

    def view_atlas(self):
        if not self._confusion_atlas:
            self.confusion_atlas()
        print(tabulate(self._confusion_frame, headers='keys', tablefmt='psql'))
        print(f"Unique ids: {self.aruco_ids}")

    def confusion_atlas(self):
        """ returns an table reprecentation for the atlas. """
        log.info("-- Calculate confusion atlas  --")
        names = self._views.keys()

        frame = dict()
        for id in self.aruco_ids:
            row = dict()
            for name in names:
                ids=self._views[name].ids
                log.info(f"Avaible ids = {ids} for name = {name}")
                if id in ids :
                    row[name] = 1
                else:
                    row[name] = 0
            frame[id]=row

        self._confusion_atlas = True
        self._confusion_frame = pd.DataFrame(frame)

    def build(self):
        """ Builds the dependency tre for the atlas projection"""
        if not self._confusion_atlas:
            self.confusion_atlas()
        log.info("Started building atlas")
        for id in self.aruco_ids:
            ct = corner(id)
            self.aruco_corners.append(ct)
            if id == self._aruco_origin_id:
                log.info("Origin Was set")
                self._aruco_origin = ct
        # Findex is the file name for each view
        # findex = self._confusion_frame.index
        # Is there more corners in this view?
        # if yes calculate path to origin.
        # store connectino between origin and next view
        # if no next view
        # Nex view loaded:
        # for every aruco calucate ptahs between.
        # are the any previously known arucos?
        # if yes calulate path to aruco and store that to the new one.
        # if no add aruco to list.
        # reppet untill out of views.
        # test if every aruco have connection to origin
        # if no WARN and delet the unconnected views
        pass











