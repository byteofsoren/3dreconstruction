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

# Atlas dependent classes import
from atlasCL_Corner import Corner
from atlasCL_view   import View
from atlasCL_Transfer import Transfer


# == Logging basic setup ===
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/atlas.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===




# == Atlas START ==
class Atlas():
    """
    The map object calculats the relations between view and aruco corners

    :param list setconf: is a link to the set.yaml
    """
    _corner_proj = dict()
    # _views = list()

    _views = dict()
    """A dict of views in in the atlas indexed by name of the view"""
    aruco_ids:list=list()
    """
    Contains a list of aruco corner ids
    identified in the set
    """
    corners:dict = dict()
    """Stores the aruco corners in the atlas"""
    _aruco_origin_id:int = 0
    """origin id in the set readset loaded. The setting is in set.yaml for that set"""
    _aruco_origin:Corner
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


    def add_View(self, view):
        """
            Add a view to the map

            :param view: is a link to the view object
        """
        print(f"atlas add_View {view.name}")
        log.info(f"#Atlas add view {view}")
        self._views[view.name] = view
        if not view.ids is None:
            for index in view.ids:
                if not index in self.aruco_ids:
                    log.info(f"#add {index}")
                    self.aruco_ids.append(index)

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
        print("atlas confusion_atlas")
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
        print("atlas build")
        if not self._confusion_atlas:
            self.confusion_atlas()
        log.info("Started building atlas")
        """ Create corners with Id is done here """
        for id in self.aruco_ids:
            # Create a corner with a link to the atlas
            # ------------
            self.corners[id]=Corner(id,self) # Dict of corner class
            # -- Stored --
            if id == self._aruco_origin_id:
                log.info("Origin Was set")
                self._aruco_origin = self.corners[id]
                self.corners[id].aruco_value = 0
        frame = self._confusion_frame
        rows=frame.index
        cols = frame.keys()
        """ Atlas: Connecting the corners with each view is done here """
        for name in self._views.keys():
            ids = self._views[name].ids
            config = self._views[name].corners # dict of {id:(tvec,rvec,corners)}
            print(config)
            # Connect corner with view over trasfer.
            for id in config.keys():
                tf = Transfer(self.corners[id],config[0],config[1],config[2])





        # for col in cols: # For every column
        #     for row in rows: # For every row
        #         if frame.loc[row][col] == 1:
        #             # Add views for that aruco corner
        #             log.info(f"#atlas:build:L3 col:{col};row:{row}")
        #             # print(f"#atlas:build:L3 col:{col};row:{row}")
        #             view_tem = self._views[row]
        #             self.corners[col].add_View(view_tem)

        # for c in self.corners.keys():
        #     print("---------------------------")
        #     localviews = self.corners[c].views
        #     print(f"id={self.corners[c].id} count:{len(localviews.keys())}")
        #     for v in localviews.keys():
        #         print(localviews[v].name)
        #     print("---------------------------")


        arkeys = [k for k in self.corners.keys()]
        """Atlas: Pruning the corners so that non connected is removed"""
        for c in arkeys:
            tlist = self.corners[c].getconnections(True)
            tcon = self.corners[c].is_connected()
            log.info(f"{c}=>{tlist}, {'OK' if tcon else '--'}")
            if not tcon:
                self.corners.pop(c)
        log.info(f"Pruning corners: {self.corners.keys()}")
        arkeys = [k for k in self.corners.keys()]
        """Atlas: Value update for connection"""
        # maxval = 10e6
        # i = 0
        # cont = True
        # maxcount = len(arkeys)*2*np.log(len(arkeys))
        # log.info(f"maxcount={maxcount}")
        # while cont :
        #     values = np.array(([self.corners[x].aruco_value for x in self.corners.keys()]))
        #     log.info(f"iteration i={i} {values} [{' maxval present' if maxval in values else ' --'}]")
        #     # If values contains any max values enter:
        #     if maxval in values:
        #         # The arkeys is a list of corner keys.
        #         for key in arkeys:
        #             # For each corner
        #             arc:corners = self.corners[key]
        #             log.info(f"Searching key={key} child {arc.aruco_value} {'Enter' if arc.aruco_value < maxval else 'skip'}")
        #             # If the corner is less then max value enter.
        #             if arc.aruco_value < maxval:
        #                 con = arc.connection
        #                 log.info(f"Corner id: {arc.id} conkeys:{con.keys()}")
        #                 # For each connection from previous corner to a next corner
        #                 for conkey in con.keys():
        #                     v = con[conkey].obj_link.aruco_value
        #                     log.info(f"v={v} {'OK' if not v < maxval else '--'}")
        #                     if v ==  maxval:
        #                         log.info(f"Update child {conkey}")
        #                         con[conkey].obj_link.aruco_value = arc.aruco_value + 1
        #     else:
        #         log.info("Solution found")
        #         cont = False # A solution must have bean found if no maxvals in array is found.
        #     if i > maxcount:
        #         log.warn("Not al solutions was found in time")
        #         break
        #     i += 1

        """Dijkstra the shortest path to original"""
        minaruco = lambda d: min([d[key].aruco_value for key in d.keys()])
        # Finds the smallest aruco_value.

        for key in arkeys:
            # log.info(f"key:{key}, aruco_value:{self.aruco_corners[key].aruco_value}")
            # log.info(f"self.aruco_corners[key].connection={self.aruco_corners[key].connection}")
            arc:Corner = self.aruco_corners[key]
            value = arc.aruco_value
            con = arc.connection.keys()
            tmp = []
            minimal_value_index = tmp



# == Atlas END ==


def _view_test():
    """Test function for view"""
    ar = aruco.Dictionary_get(aruco.DICT_6X6_250)
    param = aruco.DetectorParameters_create()
    # fname="../datasets/P2/images/093440.jpg"   # ids[4,3,8,9,11]
    fname="../datasets/P2/images/093428.jpg"   # ids[0,4,5,6,7,9,11]
    img = cv2.imread(fname)
    cam = camera("mobile")
    cam.read_param()
    v = view(fname,img,ar,param,171/1000,cam)
    print(v)
    print("Show image")
    cv2.imwrite("../logs/temp.jpg",v.img)
    tmen = TerminalMenu(['No','Yes'])
    if tmen.show() == 1:
        print("Showing image")
        scale_percent = 40 # percent of original size
        width =  int(v.img.shape[1] * scale_percent / 100)
        height = int(v.img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(v.img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow(fname,resized)
        cv2.waitKey(0)
    else:
        print("Did not show menu")

def _view_file_select():
    """Test function to pair two views."""
    ar = aruco.Dictionary_get(aruco.DICT_6X6_250)
    param = aruco.DetectorParameters_create()
    path = Path("../datasets/P2/images/")
    fnames = list()
    for file in path.glob("*.jpg"):
        fnames.append(str(file))
    cam = camera("mobile")
    cam.read_param()
    tmen = TerminalMenu(fnames)
    s = tmen.show()
    img = cv2.imread(fnames[s])
    v = view(fnames[s],img,ar,param,171/1000,cam)
    cv2.imwrite("../logs/temp.jpg",v.img)
    scale_percent = 40 # percent of original size
    width =  int(v.img.shape[1] * scale_percent / 100)
    height = int(v.img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow(str(fnames[s]),resized)
    cv2.waitKey(0)
    print(f"fnames[s]={fnames[s]}")










