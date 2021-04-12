import numpy as np
import cv2, PIL, yaml
import logging
import h5py
import copy
from glob import glob
from pathlib import Path
import pathlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
# import pandas as pd

# Local imports
from gen_aruco import caruco_board

# == Logging basic setup ===
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/camera.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===

# Local macros:
mxstr =lambda cell: np.array2string(cell, precision=2, separator=',', suppress_small=True)

class Camera():
    """
        my camera object that takes care with camera calibration and stuff.
        :param str name: Is the name and the dicectory where the camera stores its inforameton.
    """
    # Internal member variables
    _allCorners:list
    """Decected corrners."""
    _allIds:list
    """a list of aruco id."""
    _conf:list
    """Configuration from ymal file"""
    _camera_name:str
    """Name and the string for the camera path."""

    _camera_matrix = None
    """Stores the camera matrix"""

    def __init__(self, name:str):
        log.info("Start camera object")
        self._camera_name:str = name
        # Read yaml params
        self._board = caruco_board(retboard=True,retimg=False)
        # self._aruco_lib = cv2.aruco.Dictionary_get()
        self._aruco_lib = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Declaration setup:
        self._allCorners = list()
        self._allIds = list()
        with open("camera.yaml") as f:
            temp = yaml.load(f,Loader=yaml.FullLoader)
            try:
                log.info("Trying to read camera from yaml file")
                self._conf = temp["cameras"][name]
                log.info("Suscess")
            except IndexError as e:
                log.error(f"FAIL: IndexError: {e}")
                raise e
            del(temp)

    def __str__(self):
        return f"name={self._camera_name}"

    def load_img(self):
        """ Loads the images needded for the calibration """
        ftypes = self._conf['filetypes']
        ftypes = self._conf['filetypes']
        log.info(f"Loaded filetypes {ftypes}")
        log.info(f"Loaded filetypes {ftypes}")
        aruco_lib = self._aruco_lib
        aruco_lib = self._aruco_lib
        p = Path()
        p = Path()
        board = self._board
        board = self._board
        decimator = 0
        decimator = 0
        for t in ftypes:
            log.info(f"---> Reading back filetype {t}")
            log.info(f"---> Reading back filetype {t}")
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
        log.info("Done wit loading images")


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
        """
            Rectifies the input image using the calibarted camera
            :param np.ndarray img: The input image
        """
        if (not img is None):
            try:
                # ret, mtx, dist, rvecs, tvecs
                log.info("Creating local camera matrix profile")
                corp  = self._conf['corp_img']
                corps = self._conf['corp_scaling']
                mtx   = self._camera_matrix
                dist  = self._distortion_coefficients0
                rvecs = self._rotation_vectors
                tvecs = self._rotation_vectors
                log.info("Sucsess")
            except AttributeError as e:
                print("No camera matrix is defined run calibrate first.")
                log.warn(f"FAIL: No calibration has been done {e}")
                raise e
            else:
                ret = cv2.undistort(img, mtx,dist,None)
                log.info("Rectified ok")
                if corp:
                    log.info(f"Corping img from shap {img.shape}")
                    # Get img shape
                    width,height = img.shape[:2]
                    # Calculat the new start row, col
                    start_row, start_col = int(width * (1-corps)), int(width*(1-corps))
                    # Calculat the end row, col
                    end_row, end_col = int(width * (corps)), int(width*(corps))
                    # corp the img.
                    ret = img[start_row:end_row,start_col:end_col]
                    log.info(f"Corped the img to new shape {ret.shape}")
                return ret

            def save_param(self):
                """ Saves the camera params in the camera folder """
                try:
                    log.info("Create shorthand local valiables")
                    mtx   = self._camera_matrix
                    dist  = self._distortion_coefficients0
                    rvecs = self._rotation_vectors
                    tvecs = self._rotation_vectors
                    imgsize = self._imgsize
                    log.info("Sucsess: Local varibles created")
                except AttributeError as e:
                    print("No camera matrix is defined run calibrate first.")
                    log.error(f"FAIL: No calibration has been done {e}")
                    raise e
                matrixpath = Path(f"{self._conf['path']}/{self._conf['matrixfile']}")
                log.info(f"matrixpath = {str(matrixpath)}")
                matrixfile = h5py.File(matrixpath,"w")
                matrixfile.create_dataset("mtx",data=mtx)
                matrixfile.create_dataset("dist",data=dist)
                matrixfile.create_dataset("rvecs",data=rvecs)
                matrixfile.create_dataset("tvecs",data=tvecs)
                matrixfile.create_dataset("imgsize",data=imgsize)
                matrixfile.close()
                log.info("Sucsess: Matrixfile written to disk and closed")
        else:
            err = "The inupt image was None"
            log.exception(err)
            raise Exception(err)

    def read_param(self):
        """ Reads read back the matrix from disk. """
        matrixpath:pathlib.PosixPath= Path(f"{self._conf['path']}/{self._conf['matrixfile']}")
        log.info(f"Read back from {str(matrixpath)} exists {matrixpath.exists()}")
        if matrixpath.exists():
            matrixfile = h5py.File(matrixpath,"r")
            log.info(f"Opening the matrix file and creating the matrix object keys={matrixfile.keys()}")
            self._camera_matrix = matrixfile['mtx'][:]
            self._distortion_coefficients0 = matrixfile["dist"][:]
            self._rotation_vectors = matrixfile['rvecs'][:]
            self._rotation_vectors = matrixfile['tvecs'][:]
            self._imgsize = matrixfile['imgsize'][:]
            matrixfile.close()
            log.info("Params restored and matrixfile closed")
        else:
            raise Exception("File not found")

    def save_param(self):
        """ Saves the camera params in the camera folder """
        try:
            mtx   = self._camera_matrix
            dist  = self._distortion_coefficients0
            rvecs = self._rotation_vectors
            tvecs = self._rotation_vectors
            imgsize = self._imgsize
        except AttributeError as e:
            log.warn(f"No calibration has been done {e}")
            raise e
        matrixpath:pathlib.PosixPath= Path(f"{self._conf['path']}/{self._conf['matrixfile']}")
        log.info(f"Read back from {str(matrixpath)} exists {matrixpath.exists()}")
        matrixfile = h5py.File(matrixpath,"w")
        matrixfile.create_dataset(name="mtx",data=mtx)
        matrixfile.create_dataset(name="dist",data=dist)
        matrixfile.create_dataset(name="rvecs",data=rvecs)
        matrixfile.create_dataset(name="tvecs",data=tvecs)
        matrixfile.create_dataset(name="imgsize",data=imgsize)

    def ext_convert(self, T:np.ndarray, u:int,v:int)->np.ndarray:
        """ Converts the in picture u,v cordinates to real world X,Y and Z
            Cordinates usig the transfer matrix T.

            :param T: Transfer matrix 4x4.
            :param u: The x position on an image
            :param v: the y position on an image
        """
        # Unknown right now
        Pu = 1
        Pv = 1
        # Sise of half the image
        #    self._imgsize
        U0 = self._imgsize[0]/2
        V0 = self._imgsize[1]/2
        print(f"U0={U0}, V0={V0}")
        # extra to u,v
        w  = 1
        x  = np.array([[u,v,w]]).T
        print(f"x=\n{x}")
        K1 = np.array([[1/Pu,   0 , U0],
                       [   0, 1/Pv, V0],
                       [0   ,    0,  1]])
        print(K1)
        # K2 = np.ndarray([[f, 0, 0, 0],
        #                  [0, f, 0, 0],
        #                  [0, 0, 1, 0]])
        K2 = np.hstack((self._camera_matrix,np.zeros((3,1))))
        print(f"Argumented camera_matrix=\n{mxstr(K2)}")
        breakpoint()
        K = K1@K2
        print(f"K=\n{mxstr(K)}")
        Kinv = np.linalg.inv(K)
        print(f"Kiv=\n{Kinv}")
        ret = T@Kinv@x
        print(ret)
        return ret

    @property
    def mtx(self):
        if self._camera_matrix is None:
            self.read_param()
        return self._camera_matrix

    @mtx.setter
    def mtx(self,matrix):
        raise ValueError("Can not be set")

    @property
    def dist(self):
        if self._camera_matrix is None:
            self.read_param()
        return self._distortion_coefficients0

    @dist.setter
    def dist(self,matrix):
        raise ValueError("Can not be set")



def resize_img(img,scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    return  cv2.resize(img, dim, interpolation = cv2.INTER_AREA)




def main():
    # Basic setup
    with open('./camera.yaml','r') as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    tests = conf['codetests']
    log.info("Main start")
    # End basic setup start tests:
    cam = Camera("mobile")
    log.info(f"loaded {cam}")
    if tests['camera_calibration']:
        cam.load_img()
        cam.calibrate()
    if tests['save_camera']:
        log.info("Storing to file")
        cam.save_param()
        cameraMatrix_back = cam._camera_matrix
        cam.read_param()
        eq = np.allclose(cam._camera_matrix,cameraMatrix_back)
        log.info(f"Is equal? {eq}")
    if tests['rectify_img']:
        cam.read_param()
        # img = cv2.imread("../datasets/P1/original/IMG_20201124_140004.jpg")
        img = cv2.imread("../datasets/P1/images/IMG_20201124_140100.jpg")
        rimg = cam.rectify(img)
        cv2.imwrite("../logs/temp.jpg",img)
        cv2.imshow("RAW image", resize_img(img,0.4))
        cv2.waitKey(0)
        # cv2.imshow("Corrected img", rimg)
        cv2.imshow("Corrected image", resize_img(rimg,0.4))
        cv2.waitKey(0)





if __name__ == '__main__':
    main()


