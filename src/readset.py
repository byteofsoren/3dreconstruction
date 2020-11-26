import numpy as np
import cv2, yaml
import logging as log
from glob import glob
from pathlib import Path
import pathlib
from mpl_toolkits.mplot3d import Axes3D
from gen_aruco import caruco_board
import matplotlib.pyplot as plt
import matplotlib as mpl

class dataset():
    """Classa handle for datasets in this project.
    """
    _conf:list # Stores the global configuration from readset.yaml
    _setdir:pathlib.PurePath # Stores the directory for the dataset
    _setconf:list # Stores the configuration of the set it self.
    def __init__(self, name:str):
        """ Creatse the object
        input:
            name:str := Name of a sub dir in defined in readset.yaml [sets][path]
        """
        with open('./readset.yaml','r') as f:
            conf = yaml.load(f,Loader=yaml.FullLoader)
        lconf = conf['log']
        log.basicConfig(filename=lconf['file'], filemode='w', format=lconf['format'],level=lconf['level'])
        self._conf = conf['sets']
        setdir:str = self._conf['path']
        setdir:pathlib.PosixPath = Path(setdir.format(name=name))
        setfile:pathlib.PosixPath = Path(f"{setdir}/{self._conf['setfile']}")
        print(setfile)
        # Check if the direcory exsits:
        if setdir.exists() and setdir.is_dir():
            log.info(f"The given path was: {str(setdir)} and it existed")
            self._setdir = setdir
            if setfile.exists() and setfile.is_file():
                log.info(f"The file {str(setfile)} existed")
                with open(str(setfile),"r") as sf:
                    self._setconf = yaml.load(sf, Loader=yaml.FullLoader)
                self._camera = self._setconf['camera']
                log.info(f"camera='{self._camera}'")
                self._camera_pose= Path(f"{setdir}/{self._setconf['cameraposes']}")
                pose_exists = self._camera_pose.exists()
                log.info(f"File used to store camera poses is '{self._camera_pose}' exists {pose_exists}")
            else:
                err = "The setfile described in readset.yaml was not found in this diertory"
                log.ERROR(err)
                raise Exception(err)
            # Read the set file for that data set if it exists.
        else:
            err = "The name was not a directory or not found."
            log.ERROR(err)
            raise Exception(err)

    def calulate_camerapose(self):
        pass






def main():
    conf = None
    with open('./readset.yaml','r') as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    lconf = conf['log']
    print(lconf)
    log.basicConfig(filename=lconf['file'], filemode='w', format=lconf['format'],level=lconf['level'])
    # log.basicConfig(filename="../logs/readset.log", filemode='w',level=10)
    log.info("test")
    test = dataset("P1")

if __name__ == '__main__':
    # log.basicConfig(filename="../logs/readset.log", filemode='w',level=10)
    # log.info("Start")
    main()
