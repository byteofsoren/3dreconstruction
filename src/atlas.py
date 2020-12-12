import numpy as np
import cv2, yaml
import logging
import copy
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
    class T():
        """Transferobject for tranfereing between different corners"""
        T:np.ndarray # is the transfer matrix for this object

        def __init__(self, aruco_link, T):
            self._aruco_link = aruco_link
            log.info(f"Created transfer to {aruco_link.id}")
            self.T = T


    _views:list=list() # connection from the aruco to each view
                # where the aruco was observed.
    aruco_value:int = 10e6 # Connection value in reference to orgin set large in the begining
    def __init__(self, id:int, back_atlas):
        self.id:int = id
        self._back_atlas = back_atlas
        self._connection = dict() # The connection from corner to corner
        log.info(f"Created aruco corner id={id} connection:{self._connection}")

    def add_view(self, view):
        ids = np.array(view.ids)
        if (not ids is None) and (len(ids) > 1):
            self._views.append(view)
            # print(f"Corner:{self.id} Before {self._connection}")
            log.info("ids:{ids}, ")
            # As the id for each corner do not have an incremental
            # increasment from such tath corners can be addressed
            # by the id, a extra variable is needed as an indexer.
            indexer = 0
            for i in ids:
                if (i != self.id):
                    ar = self._back_atlas.aruco_corners[i]
                    # log.info(f"Connected {i}->{t.id}")
                    log.info(f"corners:{view.corners}, len: {len(view.corners)}")
                    Tarr = view.corners[indexer]
                    self._connection[i] = self.T(ar,Tarr)
                    indexer += 1
                # else:
                #     print(f"{i} == self.id={self.id}")
            log.info(f"Corner:{self.id} After {self._connection}")

    def get_connections(self, sort=False)->list:
        """ Returns a list of ids  """
        if sort:
            return sorted(self._connection.keys())
        else:
            return self._connection.keys()

    def is_connected(self)-> bool:
        """ Is the corner connected to other corners? """
        return len(self._connection) > 0







class view():
    """The view object contains the image and its connections to aruc corners"""
    corners:list = list()
    ids:None
    name:str # Often the filename of the frame/camera angle
    # isorigin:bool = False
    _origin_aruco:int = 0

    def __init__(self, name:str, img:np.ndarray, arucodict, arucoparam, corner_size,camera):
        """ The view object takes the following arguments
        @name: is teh name of the file tex img1.png
        @img: Is the acual content of the file as an ndarray
        @arucodict: Is a link to the aruco corners used in this project.
        @arucoparam: Aruco parametsers
        @corner_size: is the physical sice of the corner in meter
        @camera: I connection to the camera object.
        """
        log.info(f"-- View with name:{name} created --")
        self.name=str(name)
        self.img = img
        self.corner_size = corner_size
        self.camera = camera
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
        # tmp = aruco.estimatePoseSingleMarkers(self.corners,self.corner_size,self.camera)

        # Flatten the list incase of problem
        log.info(f"ids: {self._ids}")
        if not self._ids is None:
            self.ids = [x[0] for x in self._ids]
        else:
            raise Exception("No corner in images")

    def __str__(self):
        return f"View object filename={self.name} pressent aruco ids = {self.ids}"

    def estimate_markers(self):
        mtx = self.camera._camera_matrix
        dist = self.camera._distortion_coefficients0
        self.rvec, self.tvec= aruco.estimatePoseSingleMarkers(self.corners,self.corner_size,mtx,dist)



class atlas():
    """The map object calculats the relations between view and aruco corners"""
    _corner_proj = dict()
    # _views = list()

    _views = dict()
    aruco_ids:list=list()  # Contains a list of aruco corner ids
                    # indentified in the set
    aruco_corners:dict = dict() # Stores the aruco corners in the atlas
    _aruco_origin_id:int = 0 # origin id set in atlas.yml
    _aruco_origin:corner # A handle to the origin aruco
    _confusion_atlas=False
    _confusion_frame:pd.DataFrame # Stores the confusion frame

    def __init__(self, setconf):
        with open('./atlas.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        self._setconf = setconf
        self._aruco_origin_id = setconf['origin']


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
            # Create a corner with a link to the atlas
            self.aruco_corners[id]=corner(id,self)
            if id == self._aruco_origin_id:
                log.info("Origin Was set")
                self._aruco_origin = self.aruco_corners[id]
                self.aruco_corners[id].aruco_value = 0

        frame = self._confusion_frame
        rows=frame.index
        cols = frame.keys()
        for col in cols: # For every column
            for row in rows: # For every row
                if frame.loc[row][col] == 1:
                    # Add views for that aruco corner
                    log.info(f"#atlas:build:L3 col:{col};row:{row}")
                    view = self._views[row]
                    self.aruco_corners[col].add_view(view)

        ckeys = [k for k in self.aruco_corners.keys()]
        for c in ckeys:
            tlist = self.aruco_corners[c].get_connections(True)
            tcon = self.aruco_corners[c].is_connected()
            log.info(f"{c}=>{tlist}, {'OK' if tcon else '--'}")
            if not tcon:
                self.aruco_corners.pop(c)
        log.info(f"Pruning corners: {self.aruco_corners.keys()}")
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











