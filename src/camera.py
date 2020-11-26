import numpy as np
import cv2, PIL, yaml
import logging as log
from glob import glob
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from gen_aruco import caruco_board
import matplotlib.pyplot as plt
import matplotlib as mpl
# import pandas as pd



class camera():
    """my camera object that takes care with camera calibration and stuff."""
    _allCorners:list=list()
    _allIds:list=list()
    def __init__(self, name:str):
        self._camera_name:str = name
        # Read yaml params
        self._board = caruco_board(retboard=True,retimg=False)
        # self._aruco_lib = cv2.aruco.Dictionary_get()
        self._aruco_lib = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        log.info("Start camera object")
        with open("camera.yaml") as f:
            temp = yaml.load(f,Loader=yaml.FullLoader)
            try:
                self._conf = temp["cameras"][name]
            except IndexError as e:
                log.warn(f"IndexError: {e}")
                raise e
            del(temp)


    def load_img(self):
        """ Loads the images needded for the calibration """
        ftypes = self._conf['filetypes']
        log.info(f"Loaded filetypes {ftypes}")
        aruco_lib = self._aruco_lib
        p = Path()
        board = self._board
        decimator = 0
        for t in ftypes:
            for img in p.glob(f"../camera/{self._camera_name}/*.{t}"):
                frame = cv2.imread(str(img))
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                self._imgsize = gray.shape
                res = cv2.aruco.detectMarkers(gray, aruco_lib)

                if len(res[0]) > 0:
                    res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
                    if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                        self._allCorners.append(res2[1])
                        self._allIds.append(res2[2])
                decimator += 1


    def calibrate(self):
        """ Calibrates if the images have bean loaded in to the object. """
        imsize = self._imgsize
        log.info("Start calibration")
        cameraMatrixInit = np.array([[ 2000.,    0., imsize[0]/2.],
                                    [    0., 2000., imsize[1]/2.],
                                    [    0.,    0.,           1.]])
        distCoeffsInit = np.zeros((5,1))
        log.info("Initial camera matrix is generated")
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL)
        (ret, camera_matrix, distortion_coefficients0,
        rotation_vectors, translation_vectors,
        stdDeviationsIntrinsics, stdDeviationsExtrinsics,
        perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=self._allCorners,
                      charucoIds=self._allIds,
                      board=self._board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
        # return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors
        log.info("Camera matrix is done")
        self._camera_matrix = camera_matrix
        self._distortion_coefficients0 = distortion_coefficients0
        self._rotation_vectors = rotation_vectors
        self._translation_vectors = translation_vectors
        log.info("Storing results in camera obj")

    def rectify(self, img:np.ndarray):
        """ Rectifies the input image using the calibarted camera """
        try:
            # ret, mtx, dist, rvecs, tvecs
            mtx   = self._camera_matrix
            dist  = self._distortion_coefficients0
            rvecs = self._rotation_vectors
            tvecs = self._rotation_vectors
        except AttributeError as e:
            print("No camera matrix is defined run calibrate first.")
            log.warn(f"No calibration has been done {e}")
            raise e
        return cv2.undistort(img, mtx,dist,None)

    def save_parm(self):
        """ Saves the camera params in the camera folder """
        try:
            mtx   = self._camera_matrix
            dist  = self._distortion_coefficients0
            rvecs = self._rotation_vectors
            tvecs = self._rotation_vectors
        except AttributeError as e:
            print("No camera matrix is defined run calibrate first.")
            log.warn(f"No calibration has been done {e}")
            raise e





def main():
    with open('./camera.yaml','r') as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    lconf = conf['log']
    log.basicConfig(filename=lconf['file'], filemode='w', format=lconf['format'],level=lconf['level'])
    log.info("Main start")
    cam = camera("mobile")
    log.info("-- Camera instance created --")
    cam.load_img()
    log.info("-- Images loaded in to object --")
    cam.calibrate()
    log.info("-- Camera calibration done --")


if __name__ == '__main__':
    main()


