import numpy as np
import cv2, yaml
import logging
import copy
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from bcolor import bcolors
from tabulate import tabulate
from pathlib import Path
from cv2 import aruco
from simple_term_menu import TerminalMenu  # Used to select display options
from shapely.geometry import Polygon # Calculations of the aria
from typing import Type # Used to be able to pass var:Type[object] to a function


# Local import
from camera import Camera

# # Atlas dependent classes import
from atlasCL.Viewmod  import View
from atlasCL.Cornermod import Corner
from atlasCL.Transfermod import Transfer, Linkable

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

    views:dict
    """A dict of views in in the atlas indexed by name of the view"""
    aruco_ids:list
    """
    Contains a list of aruco corner ids
    identified in the set
    """
    corners:dict
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
        self.corners = dict()
        self.aruco_ids = list()
        self.views = dict()
        self.G = nx.DiGraph(module='atlas')


    def add_View(self, view):
        """
            Add a view to the map

            :param View view: is a link to the view object
        """
        print(f"atlas add_View {bcolors.OKCYAN}{view.name}{bcolors.END} ids {view.ids}")
        log.info(f"#Atlas add view {view}")
        self.views[view.fname] = view
        if not view.ids is None:
            for index in view.ids:
                if not index in self.aruco_ids:
                    log.info(f"#add {index}")
                    self.aruco_ids.append(index)

        if self._conf['sort_aruco']:
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
        names = self.views.keys()

        frame = dict()
        for id in self.aruco_ids:
            row = dict()
            for name in names:
                ids=self.views[name].ids
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
        log.info(f"{bcolors.INF}Started building atlas{bcolors.END}")

        G = self.G
        cnode = lambda id:  G.add_node(id,  node=Corner(id,self))
        vnode = lambda obj: G.add_node(obj.id, node=obj)
        edge = lambda s,t,tvec,rvec: G.add_edge(s.id,t.id,transfer=Transfer(s,t,tvec,rvec))

        # """ Create corners with Id is done here """
        # for id in self.aruco_ids:
        #     cnode(id)

        """ Atlas: Connecting the corners with each view is done here """
        for name in self.views.keys():
            view = self.views[name]
            vnode(view)
            log.info(f"{name} is added to graph")
            for id in view.ids:
                log.info(f"{name} is connected with {id}")
                cnode(id)

        """ Create transfer connection """
        # Generate the edges and connect the view -> corner
        # Also create a color map for the draw function
        colormap = []
        for n in list(G.nodes):
            node = G.nodes[n]['node']
            if type(node) is Corner:
                colormap.append('#76c5a1')
                s:Corner = G.nodes[id]['node']
                if s.id == self._aruco_origin_id:
                    self._aruco_origin = s

            if type(node) is View:
                colormap.append('#76b5c5')
                for id in node.ids:
                    print(f"{node.fname} is connected to {id}")
                    t = node
                    s = G.nodes[id]['node']
                    tvec,rvec,corners2d = t.get_TRvec(id)
                    edge(s,t,tvec,rvec)
        print(G.edges)

        # Connect transfer object to each edge
        for n in list(G.nodes):
            node = G.nodes[n]['node']
            if type(node) is View:
                for id in node.ids:
                    t = node
                    s = G.nodes[id]['node']
                    print(f"s={s.id}, t={t.id}")
                    # breakpoint()
                    eobj = G.edges[s.id,t.id]['transfer']
                    node.add_transfer(eobj)
                    log.info(f"Added {eobj.name} to {node.name}")

        # Connect corner -> conren with transfer
        for n in list(G.nodes):
            node = G.nodes[n]['node']
            if type(node) is View and len(node.ids) >= 2:
                ida = 0
                idb = 1
                while True:
                    # BC = AC - BC
                    # CornerA->CornerB = (CornerA->View) - (CornerB->View)
                    # Target = View
                    # Source 1 = corner.id
                    # Source 2 = corner.otherid
                    # Note:
                    # The connection from corner to corner reports view to view
                    # The problem may be that the source and target is
                    # done in the wrong order above.
                    A = G.edges[node.ids[ida],node.id]['transfer']
                    B = G.edges[node.ids[idb],node.id]['transfer']
                    Atid = A.source.id
                    Btid = B.source.id
                    # tfAB = A - B
                    # tfBA = B - A
                    tfAB = B - A
                    tfBA = A - B
                    # tfAB:Transfer = B - A
                    # tfBA = tfAB.Tinv
                    try:
                        existing = G.edges[Atid,Btid]['transfer']
                        test = existing.dist < tfAB.dist
                        teststr = f"{bcolors.SIGN.OK if test else bcolors.SIGN.FAIL}"
                        # breakpoint()
                        log.info(f"{existing.dist:.2f} < {tfAB.dist:.2f}{teststr}")
                        if test:
                            G.add_edge(Atid,Btid, transfer=tfAB)
                        del(existing)
                    except Exception as e:
                        log.info(f"{e}: {Atid}->{Btid} Was empathy")
                        G.add_edge(Atid,Btid, transfer=tfAB)

                    try:
                        existing = G.edges[Btid,Atid]['transfer']
                        test = existing.dist < tfBA.dist
                        teststr = f"{bcolors.SIGN.OK if test else bcolors.SIGN.FAIL}"
                        log.info(f"{existing.dist:.2f} < {tfBA.dist:.2f}{teststr}")
                        if test:
                            G.add_edge(Btid,Atid, transfer=tfBA)
                        del(existing)
                    except Exception as e:
                        log.info(f"{e}: {Atid}->{Btid} Was empathy")
                        G.add_edge(Btid,Atid, transfer=tfBA)

                    ida += 1
                    idb += 1
                    if idb >= len(node.ids):
                        break


        for n in list(G.nodes):
            node = G.nodes[n]['node']
            if type(node) is View:
                s = self._aruco_origin_id
                t = node.id
                # Dijkstras algorithm to calculate the path
                path = nx.dijkstra_path(G, s,t)
                back = 0
                T=np.eye(4)
                forw = 1
                if len(path) > 1:
                    while True:
                        sid = path[back]
                        tid = path[forw]
                        tf  =     G.edges[sid,tid]['transfer']
                        temp:Corner = G.nodes[tid]['node']
                        T = T@tf.T
                        temp.T=T
                        G.nodes[tid]['pos'] = list(T[:3,3])
                        back += 1
                        forw += 1
                        if forw >= len(path):
                            break

        # ------------------------------------------------------
        # Some were here there is an miss assumption on the position
        # In my implementation the aruco and camera position
        # of each object is concatenated out of the transfer
        # matrix. That is wrong. Instead the position should be
        # derived by multiplying the transfer train with a
        # homogeneous vector, i.e [0,0,0,1]^T Representing the
        # origin marker aruco 0.
        # ------------------------------------------------------

        # Some nodes have not gotten the T matrix and position.
        # Calculate pos and T for those.
        for n in list(G.nodes):
            node = G.nodes[n]
            if node.get('pos') is None:
                node = G.nodes[n]['node']
                s = self._aruco_origin_id
                t = node.id
                path = nx.dijkstra_path(G, s,t)
                back = 0
                T=np.eye(4)
                forw = 1
                if len(path) > 1:
                    while True:
                        sid = path[back]
                        tid = path[forw]
                        tf  =     G.edges[sid,tid]['transfer']
                        temp:Corner = G.nodes[tid]['node']
                        T = T@tf.T
                        temp.T=T
                        temp.pos = list(T[:3,3])
                        G.nodes[tid]['pos'] = list(T[:3,3])
                        # print(f"{sid}->{tid} => {tf.name}")
                        # print(T)
                        # breakpoint()
                        back += 1
                        forw += 1
                        if forw >= len(path):
                            break


        G.nodes[self._aruco_origin_id]['pos']=[0,0,0]
        pos=nx.get_node_attributes(G,'pos')
        self.positions = dict()
        for key in pos.keys():
            node = pos[key]
            self.positions[key] = pos[key]
            log.info(f"{key} pos:{node[0]:.2f},{node[0]:.2f}")
            pos[key] = pos[key][0:2] # 2D projection
        nx.draw(G,pos,node_color=colormap,with_labels=True,font_weight='bold')
        if self._conf['save']['saveimg']:
            plt.savefig(self._conf['save']['mapimg'])
        if self._conf['save']['showmap']:
            plt.show()
        posdata = pd.DataFrame(self.positions).transpose()
        posdata.columns = ['X','Y','Z']
        posdata.to_csv(r'../results/posdata.csv')
        print(posdata)

    def ep_solver(self, camera:Camera):
        """
            Calculate the  Epipolar geometry to
            get the real position of the feature.
            ie. [x,y] -> [X,Y,Z]
            Where x,y is in the picture and the X,Y,Z is in the real world.
            There are many ways to do that thus in

            atlas.yaml

            There are a chance to select method as follow:
                epipolar_geometry_method: 0
                    # 0 Select a camera and the nearest camera.
                    # 1 Select a camera and every camera
                    #   containing that feature.
                    #

            :param Camera camera: Is the input camera object
        """
        G = self.G
        for n in list(G.nodes):
            node:View = G.nodes[n]['node']
            if type(node) is View:
                camera.ext_convert(node.T,1,1)

    def gd_solver(self):
        """ Gaussian Distribution data Solver
            Using the derived camera position in the
            build function and the provided feature markers
            in the CSV, this function calculates the covariance matrix.
        """
        pass







    @property
    def count(self):
        return len(self.views)


# == Atlas END ==


def _view_test():
    """Test function for view"""
    ar = aruco.Dictionary_get(aruco.DICT_6X6_250)
    param = aruco.DetectorParameters_create()
    # fname="../datasets/P2/images/093440.jpg"   # ids[4,3,8,9,11]
    fname="../datasets/P2/images/093428.jpg"   # ids[0,4,5,6,7,9,11]
    img = cv2.imread(fname)
    cam = Camera("mobile")
    cam.read_param()
    v = View(fname,img,ar,param,171/1000,cam)
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
    cam = Camera("mobile")
    cam.read_param()
    tmen = TerminalMenu(fnames)
    s = tmen.show()
    img = cv2.imread(fnames[s])
    v = View(fnames[s],img,ar,param,171/1000,cam)
    cv2.imwrite("../logs/temp.jpg",v.img)
    scale_percent = 40 # percent of original size
    width =  int(v.img.shape[1] * scale_percent / 100)
    height = int(v.img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow(str(fnames[s]),resized)
    cv2.waitKey(0)
    print(f"fnames[s]={fnames[s]}")



# if __name__ == '__main__':
#     _view_file_select()







