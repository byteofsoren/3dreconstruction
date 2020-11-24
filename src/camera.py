import numpy as np
import cv2, PIL, yaml
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
# import pandas as pd

class camera():
    """my camera object that takes care with camera calibration and stuff."""
    def __init__(self, name:str):
        self._camera_name:str = name
        # Read yaml params
        with open("camera.yaml") as f:
            self._conf = yaml.load(f,Loader=yaml.FullLoader)
        print(f"conf = {self._conf}")
        if name in self._conf["cameras"]:
            self._camera_params = self._conf['cameras'][name]
        else:
            raise Exception(f'Camera {name} not in camera.yaml')
        if ("params" in self._camera_params) and ("calibated" in self._camera_params['params']):
            # Camera is calibrated.
            pass
        elif ("params" in self._camera_params):
            # camera is not calibrated but parmas exists.
            pass
        else:
            pass

    def calibrate(self):
        "Calibrates the camera from a input directory"
        # Read all images from calibration directory.
        pass






