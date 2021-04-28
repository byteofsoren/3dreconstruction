
# mayavi needs to be at the top
from mayavi.api import Engine
from mayavi import mlab
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface


# Local imports
from camera import Camera
from bcolor import bcolors
from atlasCL.Viewmod import View
# from .atlas import Atlas
# from .gen_aruco import caruco_board


from cv2 import aruco
import numpy as np
import cv2, yaml, PIL
import logging
import pathlib
# import copy
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import matplotlib.image as mplimg
import pylab
from matplotlib.cbook import get_sample_data
from typing import Type
from glob import glob
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import multivariate_normal



# == Logging basic setup ===
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/readset.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===



class dataset():
    """
        Classa handle for datasets in this project.
        @parm name:str := Name of a sub dir in defined in readset.yaml [sets][path]
    """
    conf:list
    """Stores the global configuration from readset.yaml"""
    setdir:pathlib.PurePath
    """Stores the directory for the dataset"""
    setconf:list
    """Stores the configuration of the set it self."""
    _camera:Camera
    """is the camera object used to correct the images."""
    def __init__(self, name:str):
        # load configurations from yaml file
        with open('./readset.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        log.info("test")
        # path configuration
        self.conf = conf['sets']
        self._conf = conf
        # setdir:str = self.conf['path']
        setdir:pathlib.PosixPath = Path(self.conf['path'].format(name=name))

        setfile:pathlib.PosixPath = Path(f"{setdir}/{self.conf['setfile']}")
        # Aruco config
        self._aruco = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self._parameters = aruco.DetectorParameters_create()
        """Check if the direcory exsits: """
        if setdir.exists() and setdir.is_dir():
            log.info(f"The given path was: {str(setdir)} and it existed")
            #Store the set dir in the object
            self.setdir = setdir
            # This part sets up the configuration for the dataset it self.
            # Including what camera is used.
            if setfile.exists() and setfile.is_file():
                log.info(f"The file {str(setfile)} existed")
                with open(str(setfile),"r") as sf:
                    self.setconf = yaml.load(sf, Loader=yaml.FullLoader)
                # Create a camera object
                self._camera = Camera(self.setconf['camera'])
                self._camera.read_param()
                log.info(f"camera='{self._camera}'")

            else:
                err = "The setfile described in readset.yaml was not found in this diertory"
                log.ERROR(err)
                raise Exception(err)
            # Read the set file for that data set if it exists.
        else:
            err = "The name was not a directory or not found."
            log.ERROR(err)
            raise Exception(err)
        # try:
        #     self._atlas = atlas.atlas(self.setconf)
        # except Exception as e:
        #     raise e

    def load_anatations(self):
        setdir:str = self.setdir
        atconf = self.setconf['anatations']
        bc = lambda x: f"{bcolors.INF}{x}{bcolors.END}"
        log.info(f"length = {bc(len(atconf))}")
        data = {}
        for csvset in atconf:
            csvpath = Path(f"{setdir}/{csvset['path']}")
            csvname = csvset['name']
            csvcols = csvset['column_names']
            csvtype = csvset['type']
            csvdeli = csvset['delimiter']
            log.info(f"\nname:{bc(csvname)}\ncsvtype:{bc(csvtype)}\npath:{bc(str(csvpath))}\ncolumn names:\n\t{', '.join(csvcols)} ")
            if csvpath.exists() and csvpath.is_file():
                log.info("CSV file found")
                # Read the csv file with pandas.
                csvfile = pd.read_csv(
                        str(csvpath),
                        delimiter=csvdeli,
                        names=csvcols,
                        ) # Removed the index_col to make a merged df
                data[str(csvname)] = csvfile
                data[str(csvname)]['label'] = data[str(csvname)]['label'].str.title()
            else:
                t = f"file: {str(csvpath)} does {bcolors.WARN}NOT exist{bcolors.END}."
                log.info(t)
                raise ValueError(t)
        [data[key].insert(0,"user",key,True) for key in data.keys()]
        self.indata = pd.concat([data[key] for key in data.keys()])

    def select_data(self, andarg:dict, rest_col=[]):
        """
        Does a SQL like selection based on the dict argumnt
        Supose
        andarg={'filename':'092614.jpg','User','User1'}
        This is equal to:
            SELECT * FROM data
            WHERE filename == 092614.jpg AND User == User1

        The argument rest_col select two columns form the set
        df[['filename', 'user']]

        :param dict andarg: Dict that selects rows in df
        :param list rest_col: List for selecting columns
        """
        df = self.indata
        ret = df.loc[np.all(df[list(andarg)] == pd.Series(andarg), axis=1)]
        if not len(rest_col) > 0:
            return ret
        else:
            return ret[rest_col]


    def analyse_feature(self, andarg, rest_col=[]):
        dd = self.select_data(andarg, rest_col)
        med:pd.Series    = dd.median()
        std:pd.Series    = dd.std()
        cov:pd.DataFrame = dd.cov()
        log.info(f"median:\n{med}\nstd:\n{std}\ncovariance:\n{cov}")
        return med,std,cov


    def plot3dstats(self, andarg,  alpha=0.0):
        """
        plot3d plots the stastics for a given data frame.

        :param pd.DataFrame df: The given data frame to show stats for.
        :param str image: Is the image to plot at the bottom.
        """
        #"""
        conf = self._conf['3dplot']
        med,std,cov = self.analyse_feature(andarg, ['u','v'])
        print(1)
        # First make the image
        df = self.select_data(andarg)
        # fig = plt.Figure(figsize=(15,15))
        fig = mlab.figure()
        if conf['showimg']:
            if 'filename' in andarg.keys():
                setdir = self.setdir
                image_path = f"{setdir}/images/{df['filename'].iloc[0]}"
                # Read the color image
                img:np.ndarray = cv2.imread(image_path)
                print(img.shape)

                # Resize the image
                # scale_percent = conf['resize']
                # width = int(img.shape[1] * scale_percent / 1000)
                # height = int(img.shape[0] * scale_percent / 1000)
                # dim = (width, height)
                # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                print(img.shape)
                # Convert to gray scale
                image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(image.shape)
                obj=mlab.imshow(image)
                # obj.actor.orientation = np.rad2deg(camera.w_Rt_c.euler)
                # pp = np.array([0, 0, camera.f])[:,None]
                # w_pp = camera.w_Rt_c.forward(pp)
                # obj.actor.position = w_pp.ravel()
                obj.actor.scale = [0.8, 0.8, 0.8]
            else:
                raise IndexError("No filename was declared in andarg keys")

        # Drawing the NGD
        print(4)
        if not np.linalg.det(cov) == 0 and conf['plotstd']:
            rv = multivariate_normal(med, cov)
            zscaling = conf['zscaling']

            sample= self.select_data(andarg,['u','v'])
            imgsize = self.select_data(andarg,['width','height'])

            # Bounds parameters
            x_abs = np.max(imgsize.iloc[:,0])
            y_abs = np.max(imgsize.iloc[:,1])
            try:
                obj.actor.position = [x_abs/2,y_abs/2,0]
            except NameError:
                log.info("No image showed")
            xy_abs = np.max([x_abs,y_abs])
            ax_extent = [0,xy_abs,0,xy_abs]
            ax_ranges = [0,x_abs,0,y_abs, 0,conf['zplotrange']]

            # gstep = conf['gridstep']
            gstep = conf['resize']
            xx = np.arange(0,x_abs,gstep)
            yy = np.arange(0,y_abs,gstep)
            x_grid, y_grid = np.meshgrid(xx.T,yy.T)

            pos = np.empty(x_grid.shape + (2,))
            pos[:, :, 0] = x_grid
            pos[:, :, 1] = y_grid

            levels = np.linspace(0,1,40)

            z = zscaling*rv.pdf(pos)

            # plot the surface
            grid = mlab.surf(z.T,
                    # colormap='RdY1Bu',
                    # warp_scale=0.3,
                    warp_scale='auto',
                    representation='wireframe',
                    line_width=0.5,
                    extent=ax_ranges
                    )
            # Set opacity of the surface grid
            grid.actor.property.opacity = 0.5

            # Shows the outline box aruond the figure
            mlab.outline(
                    color=(0, 0, 0),
                    opacity=0.8
                    )
            # plot the scatter for collected data:

            # Showing or saving the figure
            mlab.view(azimuth=-30, distance=1e4, roll=-90)
            mlab.show()
            # mlab.savefig("../logs/combo_plot_mlab.png")






    def set_atlas(self, atlas_link):
        self._atlas = atlas_link

    def __str__(self):
        return f"Database at dir {str(self.setdir)}"

    def show_atlas(self):
        self._atlas.view_atlas()

    def create_views(self):
        """
            Creates view used in the atlas

            :raises MesurmentError: If the unit is not suported
        """
        log.info("--Create vews--")
        imgpath = Path(f"{str(self.setdir)}/{self.setconf['imgdir']}")
        exists = imgpath.exists() and imgpath.is_dir()
        log.info("imgpath {imgpath} exists {exists}")
        for imgtype in self.setconf['imgtypes']:
            for img in imgpath.glob(imgtype):
                log.info(f"Reading image {img}")
                frame = cv2.imread(str(img))
                rectframe = self._camera.rectify(frame)
                if self.setconf['arucosize'][1] == 'mm':
                    ars = self.setconf['arucosize'][0]/1000
                elif self.setconf['arucosize'][1] == 'M':
                    ars = self.setconf['arucosize'][0]
                else:
                    log.error("MesurmentError: The unit given ({self.setconf[1]}) is not suported")
                    raise Exception("MesurmentError")
                cam = self._camera
                log.info(f'Arucosize:{ars}, cam:{cam}')
                # v = atlas.view(img.name,rectframe, self._aruco, self._parameters, ars, cam)
                # self._atlas.add_view(v)
                try:
                    # breakpoint()
                    view = View(img.name,rectframe, self._aruco, self._parameters, ars, cam)
                except Exception as e:
                    log.info(f"view not added {e}")
                else:
                    self._atlas.add_View(view)
                    del(view)


    def build_atlas(self):
        """Shorthand for building the atlas"""
        self._atlas.build()

    def geometry_solver(self):
        self._atlas.ep_solver(self._camera)
        self._atlas.gd_solver()



    @property
    def count(self):
        return self._atlas.count



def test_set(name):
    """
        Tests a set in the dataset directory.
        :param name:str is the name of the dataset that is loaded with this function
    """
    # conf = None
    # with open('./readset.yaml','r') as f:
    #     conf = yaml.load(f,Loader=yaml.FullLoader)
    log.info("test")
    datap1 = dataset(name)
    myatlas = atlas.Atlas(datap1.setconf)
    datap1.set_atlas(myatlas)
    log.info(f"Created dataset object p1 {datap1}")
    datap1.create_views()
    log.info("Done. Loaded all images")
    datap1.show_atlas()
    log.info("builds the atlas")
    datap1.build_atlas()


def test_stats(name):
    log.info(f"Test {name}")
    data = dataset(name)
    data.load_anatations()
    # a = {'filename':'093614.jpg','label':'Nose'}
    a = {'filename':'093614.jpg'}
    # print(data.select_data(a))
    # print("--------------")
    # a = {'filename':'093614.jpg','user':'User1','label':'Nose'}
    # a = {'label':'Nose'}
    # print(data.select_data(a))
    data.plot3dstats(a)





if __name__ == '__main__':
    # test_set("P2")
    test_stats("P2")
