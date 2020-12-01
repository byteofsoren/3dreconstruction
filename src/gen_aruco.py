import numpy as np
import cv2, PIL
import yaml
import logging
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
# import pandas as pd
from pathlib import Path

# == Logging basic setup ===
log = logging.getLogger(__name__)
f_log = logging.FileHandler("../logs/gen_aruco.log")
f_log.setLevel(logging.INFO)
f_logformat = logging.Formatter("%(name)s:%(levelname)s:%(lineno)s-> %(message)s")
f_log.setFormatter(f_logformat)
log.addHandler(f_log)
log.setLevel(logging.INFO)
# == END ===



class argen():
    """Aruco corner"""
    def __init__(self, idi:int, size:int=700,bgsize:int=1):
        self._id = idi
        self._dic = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        img = np.zeros((size,size), dtype=np.uint8)
        self._img = aruco.drawMarker(self._dic,idi,size,img, borderBits=bgsize)

    def show(self):
        wn = f"Aruco nr {self._id}"
        cv2.imshow(wn,self._img)
        cv2.waitKey(0)
        cv2.destroyWindow(wn)

    def save(self, fname:str):
        log.info(f"Ar save {fname}")
        cv2.imwrite(fname,self._img)

# class caruco_board():
#     """Creates the charuco bord for camera calibration"""
#     def __init__(self,fig:plt.Figure):
#         fig.clf()
#         with open("./gen_aruco.yaml") as y:
#             conf = yaml.load(y,Loader=yaml.FullLoader)
#         self._conf = conf['charuco']
#         nx,ny = self._conf['shape']

def caruco_board(retboard:bool=False,retimg:bool=True)->np.ndarray:
    """ Creates a caruco board with the config denoted in gen_aruco.yaml
        input: None
        Return: image:nd.ndarray
    """
    log.info("Starting caruco board")
    with open("./gen_aruco.yaml") as y:
        conf = yaml.load(y,Loader=yaml.FullLoader)
    conf = conf['charuco']
    nx,ny = conf['shape']
    arucotemp = aruco.Dictionary_get(aruco.DICT_6X6_250)
    bord = aruco.CharucoBoard_create(nx,ny,1,conf['border'],arucotemp)
    if (retboard) and (retimg):
        img = bord.draw((conf['pix'],conf['pix']))
        return img,bord
    elif (retboard) and (not retimg):
        return bord
    elif (not retboard) and (retimg):
        img = bord.draw((conf['pix'],conf['pix']))
        return img
    elif (not retboard) and (not retimg):
        raise Exception("Both retboard and retimg can not be False")


def main():
    with open("./gen_aruco.yaml") as y:
        conf = yaml.load(y,Loader=yaml.FullLoader)
    with open("../aruco/gen.tex", 'w') as f:
        f.write("")

    log.info("Starting main")
    fname = conf['filename']
    path = Path(conf['path'])
    s  = conf['size']['ar']
    bs = conf['size']['bg']
    gen_img = bool(conf['images'])
    # print(gen_img)
    # print(conf)
    if path.exists() and path.is_dir() and gen_img:
        log.info("Start creating images and LaTeX")
        with open("../aruco/gen.tex", 'a') as f:
            for nr in conf['generate']:
                # print(f.format(nr=nr))
                ar = argen(nr,size=s,bgsize=bs)
                print(f"Writing: {path}/{fname.format(nr=nr)}")
                # open gen.tex to write latex
                st = str("\includegraphics[width=\\textwidth, keepaspectratio]{" + fname.format(nr=nr) +"}\n")
                f.write("\\vspace{5cm}\n")
                f.write("\\begin{center}\n")
                f.write(st)
                f.write("\\end{center}\n")
                if (not conf['charuco_origin']) and (nr == int(conf['origin'])):
                    log.info("Usual aruco corner was used to create the orign")
                    # Create the image for this number
                    ar.save(f"{path}/{fname.format(nr=nr)}")
                    f.write("\\vspace{2cm}\n")
                    f.write(str("\includegraphics[width=3cm, keepaspectratio]{origin_fig.png}\n"))
                elif (conf['charuco_origin']) and (nr == int(conf['origin'])):
                    # Create origin using a carucoboard.
                    log.info("Using the caruco board for origin")
                    imgtmp = caruco_board()
                    cv2.imwrite(f"{path}/{fname.format(nr=nr)}", imgtmp)
                    f.write("\\vspace{2cm}\n")
                    f.write(str("\includegraphics[width=3cm, keepaspectratio]{origin_fig.png}\n"))
                else:
                    ar.save(f"{path}/{fname.format(nr=nr)}")
                f.write("\\newpage\n")
                if conf['dual']:
                    log.info("dual sided paper set to true")
                    f.write("\\myemptypage\n")
            if not conf['charuco_origin']:
                log.info("Caruco borad was not used as origin thus adding it att the end")
                f.write("\\vspace{5cm}\n")
                f.write("\\begin{center}\n")
                imgtmp = caruco_board()
                cv2.imwrite(f"{path}/caruco_board.png", imgtmp)
                f.write(str("\includegraphics[width=\\textwidth, keepaspectratio]{caruco_board.png}\n"))
                f.write("\\end{center}\n")
                f.write("\\newpage\n")
                if conf['dual']:
                    log.info("dual sided paper set to true")
                    f.write("\\myemptypage\n")



    print("No goto ../latex and run $ latexmk -pdf main to generate the  pdf")

def gen():
    b = caruco_board()
    b.show()


if __name__ == '__main__':
    with open('./gen_aruco.yaml','r') as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    del(conf)
    main()
    # gen()
    log.info("END of programm")

