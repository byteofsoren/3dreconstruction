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
    """
    Class for aruco corner relations
    :param int id: Is the id of the corner
    :param atlas back_atlas: is a link back to the atlas object.
    """
    class T():
        """Transfer object for transferring between different corners"""
        T:np.ndarray # is the transfer matrix for this object
        """Transfer matrix for this link"""

        aruco_link:None
        """Connection to corner"""

        def __init__(self, aruco_link, T):
            self.aruco_link = aruco_link
            log.info(f"Created transfer to {aruco_link.id}")
            self.T = T


    _views:list=list()
    """Connection from the aruco to each view  where the aruco was observed. """
    aruco_value:int = 10e6 #
    """Connection value in reference to origin set large in the beginning"""

    def __init__(self, id:int, back_atlas):
        self.id:int = id
        self._back_atlas = back_atlas
        self.connection = dict() # The connection from corner to corner
        log.info(f"Created aruco corner id={id} connection:{self.connection}")

    def add_view(self, view):
        """ Adds a view to a corner
        :param view a view class pointer
        """
        ids = np.array(view.ids)
        if (not ids is None) and (len(ids) > 1):
            self._views.append(view)
            # print(f"Corner:{self.id} Before {self.connection}")
            log.info("ids:{ids}, ")
            """
                As the id for each corner do not have an incremental
                increment from such that corners can be addressed
                by the id, a extra variable is needed as an indexer.
            """
            indexer = 0
            for i in ids:
                if (i != self.id):
                    ar = self._back_atlas.aruco_corners[i]
                    # log.info(f"Connected {i}->{t.id}")
                    log.info(f"corners:{view.corners}, len: {len(view.corners)}")
                    Tarr = view.corners[indexer]
                    self.connection[i] = self.T(ar,Tarr)
                    indexer += 1
                # else:
                #     print(f"{i} == self.id={self.id}")
            log.info(f"Corner:{self.id} After {self.connection}")

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






class view():
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
    corners:list = list()
    """Corners is a list of corner objects in the view"""
    ids:None
    """IDS is the list of ids in the view"""
    name:str # Often the filename of the frame/camera angle
    """Name is often the filename of the view"""
    _origin_aruco:int = 0


    def __init__(self, name:str, img:np.ndarray, arucodict, arucoparam, corner_size,camera):
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

        # Flatten the list incase of problem
        log.info(f"ids: {self._ids}")
        if not self._ids is None:
            self.ids = [x[0] for x in self._ids]
        else:
            raise Exception("CornerERROR")

    def __str__(self):
        return f"View object filename={self.name} pressent aruco ids = {self.ids}"

    def estimate_markers(self):
        """ Estimates the marker position in the view and store them as rvec and tvec parameters """
        mtx = self.camera._camera_matrix
        dist = self.camera._distortion_coefficients0
        # log.info(f"mtx=\{mtx}\ndist=\n{dist}")
        for c in self.corners:
            log.info(f"\n{c}")
        log.info("size={self.corner_size}M")
        self.rvec, self.tvec, _objPoints = aruco.estimatePoseSingleMarkers(
                corners=self.corners,
                markerLength=self.corner_size,
                cameraMatrix=mtx,
                distCoeffs=dist)
        log.info(f"<{self.name}>\trvec={self.rvec.shape}\ttvec={self.tvec.shape}")



class atlas():
    """The map object calculats the relations between view and aruco corners
    :param list setconf is a link to the set.yaml"""
    _corner_proj = dict()
    # _views = list()

    _views = dict()
    """A dict of views in in the atlas indexed by name of the view"""
    aruco_ids:list=list()
    """
    Contains a list of aruco corner ids
    identified in the set
    """
    aruco_corners:dict = dict()
    """Stores the aruco corners in the atlas"""
    _aruco_origin_id:int = 0
    """origin id in the set readset loaded. The setting is in set.yaml for that set"""
    _aruco_origin:corner
    """A handle to the origin aruco"""
    _confusion_atlas=False
    """Checks to see if the confusion frame was calulated"""
    _confusion_frame:pd.DataFrame # Stores the confusion frame
    """Panda dataframe containing with frame contains what corner"""
    _aruco_origin=None
    """A reference to the aruco origin"""

    def __init__(self, setconf):
        with open('./atlas.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        self._conf = conf
        self._setconf = setconf
        self._aruco_origin_id = setconf['origin']


    def add_view(self, view):
        """
            Add a view to the map
            :param view view is a link to the view object
        """
        log.info(f"#Atlas add view {view}")
        self._views[view.name] = view
        if not view.ids is None:
            for index in view.ids:
                if not index in self.aruco_ids:
                    log.info(f"#add {index}")
                    self.aruco_ids.append(index)
                if self._conf['add_view_estimatemarkers']:
                    view.estimate_markers()


        self.aruco_ids = sorted(self.aruco_ids)

    def view_atlas(self):
        """Print the confusion array to terminal"""
        if not self._confusion_atlas:
            self.confusion_atlas()
        print(tabulate(self._confusion_frame, headers='keys', tablefmt='psql'))
        print(f"Unique ids: {self.aruco_ids}")

    def confusion_atlas(self):
        """
            Calculates the confusion array for with corner is connected
            to with view and store it as an panda.DataFrame
        """
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
        """ Builds the dependency tree for the atlas projection"""
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

        arkeys = [k for k in self.aruco_corners.keys()]
        """Pruning the corners so that non connected is removed"""
        for c in arkeys:
            tlist = self.aruco_corners[c].getconnections(True)
            tcon = self.aruco_corners[c].is_connected()
            log.info(f"{c}=>{tlist}, {'OK' if tcon else '--'}")
            if not tcon:
                self.aruco_corners.pop(c)
        log.info(f"Pruning corners: {self.aruco_corners.keys()}")
        arkeys = [k for k in self.aruco_corners.keys()]
        """Value update for connection"""
        maxval = 10e6
        i = 0
        cont = True
        while cont :
            values = np.array(([self.aruco_corners[x].aruco_value for x in self.aruco_corners.keys()]))
            log.info(f"iteration i={i} {values} [{' maxval present' if maxval in values else ' --'}]")
            # If values contains any max values enter:
            if maxval in values:
                # The arkeys is a list of corner keys.
                for key in arkeys:
                    # For each corner
                    arc:corners = self.aruco_corners[key]
                    log.info(f"Searching key={key} child {arc.aruco_value} {'Enter' if arc.aruco_value < maxval else 'skip'}")
                    # If the corner is less then max value enter.
                    if arc.aruco_value < maxval:
                        con = arc.connection
                        log.info(f"Corner id: {arc.id} conkeys:{con.keys()}")
                        # For each connection from previous corner to a next corner
                        for conkey in con.keys():
                            v = con[conkey].aruco_link.aruco_value
                            log.info(f"v={v} {'OK' if not v < maxval else '--'}")
                            if v ==  maxval:
                                log.info(f"Update child {conkey}")
                                con[conkey].aruco_link.aruco_value = arc.aruco_value + 1
            else:
                log.info("Solution found")
                cont = False # A solution must have bean found if no maxvals in array is found.
            if i > len(arkeys)*2*np.log(len(arkeys)):
                log.warn("Not al solutions was found in time")
                break
            i += 1

        """Dijkstra the shortest path to origin"""
        for key in arkeys:
            log.info(f"key:{key}, aruco_value:{self.aruco_corners[key].aruco_value}")


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











