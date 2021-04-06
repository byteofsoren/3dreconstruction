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
# from .camera import camera

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

            :param view: is a link to the view object
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
                    tfAB = A - B
                    tfBA = B - A
                    G.add_edge(Atid,Btid, transfer=tfAB)
                    G.add_edge(Btid,Atid, transfer=tfBA)
                    # print(f"A={A.name}\nB={B.name}")

                    ida += 1
                    idb += 1
                    if idb >= len(node.ids):
                        break


        for n in list(G.nodes):
            node = G.nodes[n]['node']
            # breakpoint()
            if type(node) is View:
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
                        G.nodes[tid]['pos'] = list(T[:2,3])
                        back += 1
                        forw += 1
                        if forw >= len(path):
                            break

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
                        G.nodes[tid]['pos'] = list(T[:2,3])
                        # print(f"{sid}->{tid} => {tf.name}")
                        # print(T)
                        # breakpoint()
                        back += 1
                        forw += 1
                        if forw >= len(path):
                            break


        G.nodes[self._aruco_origin_id]['pos']=[0,0]
        pos=nx.get_node_attributes(G,'pos')
        for key in pos.keys():
            node = pos[key]
            print(f"{key} pos:{node[0]:.2f},{node[0]:.2f}")
        nx.draw(G,pos,node_color=colormap,with_labels=True,font_weight='bold')
        plt.savefig(self._conf['save']['mapimg'])
        plt.show()





        # """Dijkstra values the shortest path to original"""
        # counter = 0
        # value = 0
        # nrc = len(self.corners)
        # cop = self._aruco_origin
        # while counter < nrc*10:
        #     if cop.aruco_value < 10e5:
        #         for tf in cop.transfers:
        #             tf.target.aruco_value = min(cop.aruco_value + 1, tf.target.aruco_value)
        #     fuck = True
        #     while fuck:
        #         rnd = np.random.random_integers(0,nrc)
        #         try:
        #             key = list(self.corners.keys())[rnd]
        #             fuck = False
        #         except IndexError as e:
        #             log.warn(f"random vale {rnd} has no key")
        #             pass
        #     # log.info(f"nrc={nrc}, rnd={rnd}, key={key}")
        #     cop = self.corners[key]
        #     counter += 1

        # """ log aruco values  """
        # for corner in self.corners.values():
        #     val = f"{f'{bcolors.OK}{corner.aruco_value} [GOOD]{bcolors.END}' if corner.aruco_value < 10e5 else f'{bcolors.ERR}  [BAAD]{bcolors.END}'}"
        #     # log.info(f"{corner.name} -> val:{corner.aruco_value}")
        #     log.info(f"{corner.name} -> val:\t{val}")

        # """ Pruning of connections """
        # # As corners and perhaps views has disappeared from the set,
        # # a pruning of transfers must be done.
        # # c1.transfers -> tf[s:c1 t:c2] -> c2
        # # c1.transfers -> tf[s:c1 t:c3]
        # # last must be removed both from transfers
        # # because c3 do not exist.
        # onedict = self.corners.copy()
        # onedict.update(self.views)
        # for fkey in onedict.keys():
        #     transfers = list()
        #     log.info(f"-- {fkey} ---")
        #     for tf in onedict[fkey].transfers:
        #         if tf.target in onedict.values() and tf.source in onedict.values():
        #             log.info(f"\t|- append tf:{bcolors.OKBLUE}{tf}{bcolors.END}")
        #             transfers.append(tf)
        #         else:
        #             log.warn(f"\t|- destroyed tf:{bcolors.WARN}{tf}{bcolors.END}")
        #     if len(onedict[fkey].transfers) == len(transfers):
        #         log.info(f"{bcolors.SIGN.OK} transfers equal")
        #     else:
        #         # breakpoint()
        #         diff = len(onedict[fkey].transfers) - len(transfers)
        #         log.info(f"{bcolors.SIGN.FAIL} transfers NOT equal, elements removed  {diff} new length {len(transfers)}")
        #         # breakpoint()
        #         # Store the shorted transfer
        #         onedict[fkey].transfers = transfers

        # """ Pruning unconnected corners """
        # # Due to the pruning above a few corners lost all connections.
        # # Thus they need to be pruned from the list

        # """ Drawing a networkx diagram """



        # """ Calculate distance to cameras (finaly) """
        # for view in self.views.values():
        #     # breakpoint()
        #     view.generate_transfer()
            # # find minimal transfer for view
            # log.info(f"--{bcolors.OKCYAN} {view} {bcolors.END}--")
            # mkdikt=dict()
            # tf:Transfer
            # for tf in view.transfers:
            #     log.info(f"{bcolors.INF}{tf.name}{bcolors.END} types\n\t{type(tf.source)}:{type(tf.target)}")
            #     if type(tf.target) is Corner:
            #         log.info(f"target: {tf.target.name} as value {tf.target.aruco_value}")
            #     if type(tf.source) is Corner:
            #         log.info(f"source: {tf.source.name} as value {tf.source.aruco_value}")
            #     if not type(tf.source) is None:
            #         mkdikt[tf.source.aruco_value] = tf
            #     else:
            #         breakpoint()
            # log.info(mkdikt.keys())
            # if len(mkdikt) > 0:
            #     minkey = min(list(mkdikt))
            #     log.info(f"minkey:{minkey}->{mkdikt[minkey].name} mkdikt:{mkdikt} ")
            # else:
            #     log.warn("No good key found")
            # while True:
            #     # Prune on empty dict

            #     pass



        # todo: transfers is calulated origin -> view.
        # but need to be derived view -> origin.
        # thus view need a back trace stack as a list().
        # [transefr(corner13,view3),transfer(corner5,corner13), transfer(corner0,corner5)]
        # Remember that corner -> view is the only direction for tf connecting corners.
        # And perhaps corner <--> corner exist but it can be corner --> corner only.
        # thus the back trace need to compensate for that.
        # perhaps then [view,corner,corner,corner] is a better backtrace.





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
    cam = camera("mobile")
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
    cam = camera("mobile")
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







