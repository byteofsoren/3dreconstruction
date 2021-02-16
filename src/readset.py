import numpy as np
import cv2, yaml
from cv2 import aruco
import logging
from glob import glob
from pathlib import Path
import pathlib
from mpl_toolkits.mplot3d import Axes3D
from gen_aruco import caruco_board
import matplotlib.pyplot as plt
import matplotlib as mpl
# Import the camera.
from camera import camera
# Import map classes.
import atlas

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
    _setdir:pathlib.PurePath
    """Stores the directory for the dataset"""
    setconf:list
    """Stores the configuration of the set it self."""
    _camera:camera
    """is the camera object used to correct the images."""
    def __init__(self, name:str):
        # load configurations from yaml file
        with open('./readset.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        log.info("test")
        # path configuration
        self.conf = conf['sets']
        setdir:str = self.conf['path']
        setdir:pathlib.PosixPath = Path(setdir.format(name=name))
        setfile:pathlib.PosixPath = Path(f"{setdir}/{self.conf['setfile']}")
        # Aruco config
        self._aruco = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self._parameters = aruco.DetectorParameters_create()
        """Check if the direcory exsits: """
        if setdir.exists() and setdir.is_dir():
            log.info(f"The given path was: {str(setdir)} and it existed")
            #Store the set dir in the object
            self._setdir = setdir
            # This part sets up the configuration for the dataset it self.
            # Including what camera is used.
            if setfile.exists() and setfile.is_file():
                log.info(f"The file {str(setfile)} existed")
                with open(str(setfile),"r") as sf:
                    self.setconf = yaml.load(sf, Loader=yaml.FullLoader)
                # Create a camera object
                self._camera = camera(self.setconf['camera'])
                self._camera.read_param()
                log.info(f"camera='{self._camera}'")
                # Load path to csv file
                self._camera_pose= Path(f"{setdir}/{self.setconf['cameraposes']}")
                pose_exists = self._camera_pose.exists()
                log.info(f"File used to store camera poses is '{self._camera_pose}' exists {pose_exists}")
                if not pose_exists:
                    # create the csv file
                    log.error("Pose file do not exists")
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

    def set_atlas(self, atlas_link):
        self._atlas = atlas_link

    def __str__(self):
        return f"Database at dir {self._setdir}"

    def show_atlas(self):
        self._atlas.view_atlas()

    def create_views(self):
        """
            Creates view used in the atlas
            :raises MesurmentError: If the unit is not suported
        """
        log.info("--Create vews--")
        imgpath = Path(f"{self._setdir}/{self.setconf['imgdir']}")
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
                    v = atlas.View(img.name,rectframe, self._aruco, self._parameters, ars, cam)
                except Exception as e:
                    log.info("view not added")
                else:
                    self._atlas.add_View(v)


    def build_atlas(self):
        """Shorthand for building the atlas"""
        self._atlas.build()



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


if __name__ == '__main__':
    test_set("P2")
