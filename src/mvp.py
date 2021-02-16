import numpy as np
import cv2, yaml
from tabulate import tabulate
from cv2 import aruco

from camera import camera

def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    print(f"R={R}, shape={R.shape}")
    R = np.matrix(R).T
    invTvec = R@np.matrix(-tvec)
    invRvec, _ = cv2.Rodrigues(R)


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """ Get relative position ofor rvec2, tvec2 copose te return rvec, tvec """
    rvec1, tvec1 = rvec1.reshape((3,1)), tvec1.reshape((3,1))
    rvec2, tvec2 = rvec2.reshape((3,1)), tvec2.reshape((3,1))
    # Inverse the secound marker
    invRvec, invTvec = inversePerspective(rvec2,tve2)
    info = cv2.composeRT(rvec1,tvec1,invRvec,invTvec)
    composeRvec, composeTvec = info[0], info[1]
    composeRvec = composeRvec.reshape((3,1))
    composeTvec = composeTvec.reshape((3,1))
    return composeRvec, composeTvec

class test_atrib:
    """Just a test class"""
    _val = 0
    def __init__(self, initial_value):
        self._val = initial_value
        print(f"self._val={self._val}")

    @property
    def value(self):
        print("setting {val}")
        return self._val

    @value.setter
    def value(self, val):
        print(f"Setting val={val}")
        self._val = val



def main():
    cam = camera("mobile")
    mtx=cam.mtx
    print(cam)
    print(mtx)
    # print(f"cam._camera_matirx={cam._camera_matrix}")
    pass

if __name__ == '__main__':
    main()


