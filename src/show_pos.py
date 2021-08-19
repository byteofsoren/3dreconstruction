import pandas as pd
import pygame, sys
import numpy as np
import random
import time
from pathlib import Path
from numpy import cos as C
from numpy import sin as S
from PIL import Image
from pygame.locals import *
from datetime import datetime



class posdata:
    """docstring for posdata"""
    tf_z = lambda self, a,dx,dy,dz: np.array([[C(a),-S(a),0,dx],[S(a),C(a),0,dy],[0,0,1,0],[0,0,0,1]])
    tf_2D = lambda self, a,dx,dy: np.array([[C(a),-S(a),dx],[S(a),C(a),dy],[0,0,1]])
    alpha = lambda self, a: a*(np.pi/180)
    def __init__(self, filename:str):
        self.fname= Path(filename)
        # Do the file exists?
        if (not self.fname.exists()) or (not self.fname.is_file()):
            raise OSError(filename)
        print("[Ok] file exits")
        # Read the data.
        self.data = pd.read_csv(str(self.fname),index_col=0)
        self.matrix = self.data[['X','Y','Z']].to_numpy()
        self.matrix2D = self.data[['X','Y']].to_numpy()

    def transform(self,scale,a,dx,dy,dz):
        points:np.ndarray = scale*self.matrix.T
        points =  np.vstack((points,np.ones((points.shape[1]))))
        res =  self.tf_z(0,dx,dy,dz)@self.tf_z(self.alpha(a),0,0,0)@points
        return pd.DataFrame(res[:-1].T, index=self.data.index)

    def transform2D(self,scale,a,dx,dy):
        inv = np.array([[1,0],[0,-1]])
        points:np.ndarray = scale*inv@self.matrix2D.T
        points =  np.vstack((points,np.ones((points.shape[1]))))
        res =  self.tf_2D(0,dx,dy)@self.tf_2D(self.alpha(a),0,0)@points
        return pd.DataFrame(res[:-1].T, index=self.data.index)

class imgs:
    """docstring for imgs"""
    camera_img:Image
    bg_img:Image
    aruco_imgs = dict()
    _aruco_size = 1.0
    def __init__(self, bgfname:str):
        bg = Path(bgfname)
        if (not bg.is_file()) or (not bg.exists()):
            raise OSError(bgfname)
        self.bg_img = Image.open(str(bg))

    def add_camera(self,fname:str):
        fn = Path(fname)
        if (not fn.is_file()) or (not fn.exists()):
            raise OSError(fn)
        im = Image.open(str(fn))
        self.camera_img = im

    def add_aruco(self, nr:int, fname:str):
        print(f"Load {fname}", end=" ")
        fn = Path(fname)
        if (not fn.is_file()) or (not fn.exists()):
            raise OSError(fn)
        im = Image.open(str(fn))
        self.aruco_imgs[nr] = im
        print("[OK]")

    def auto_aruco(self, path, data:posdata):
        for row in data.data.index:
            if row.isdigit():
                frow = str(row)
                if int(row) < 10:
                    frow = str(f"0{row}")
                try:
                    self.add_aruco(row,path.format(frow))
                except OSError as e:
                    print(path.format(row),end=" ")
                    print("Not found")




class window:
    """docstring for window"""
    window_scale = 0.4      # Scaling factor for the window
    pdata = None            # connection tho the input data
    marker_scale = 1.0      # Image size of the markers
    camera_scale = 1.0      # Image size of the camera marker
    pos = [452,330]         # Position of the data
    angle = 37               # Angle of the data
    shape_scale = 250.0     # Scaling of the data
    cross_length = 30
    cross_width  = 50
    togle_marker = True
    togle_camera = True
    savedir = Path("../results/correct_pos/")
    savePrint = False

    def __init__(self, imgobj:imgs):
        print("my window")
        self.imgs = imgobj
        shape = imgobj.bg_img.size
        # Display object is created here
        self.display = pygame.display.set_mode(list(map(lambda x: int(x*self.window_scale), shape)))
        pygame.display.set_caption("Data corrector")

        self.color = (255,255,255)
        self.display.fill(self.color)

        pygame.font.init() # you have to call this at the start,
                   # if you want to use this module.
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = self.myfont.render('Some Text', False, (0, 0, 0))

        # Reder bg
        self.cross_length = int(10*self.window_scale)
        self.cross_width  = int(5*self.window_scale)
        pygame.display.update()



    def start(self):
        run = True
        dpos = 1
        angle = 0
        while run:
            if self.savePrint:
                self.print_screen()
            self.display.fill(self.color)
            text_msg = list()
            text_msg.append(f"Pos={self.pos},dpos={dpos} exit by [esc]")
            text_msg.append(f"angle={self.angle}")
            text_msg.append(f"marker_scale={self.marker_scale}, camera_scale={self.camera_scale}")
            text_msg.append(f"shape_scale={self.shape_scale}")
            text_displace = 20
            i=0
            for msg in text_msg:
                textsurface = self.myfont.render(msg, False, (150, 150, 150))
                self.display.blit(textsurface,(0,i))
                i+= text_displace
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.exit()
                        pygame.quit()
                        sys.exit()
                    elif event.key == K_SPACE:
                        self.savePrint = True
                        for msg in text_msg:
                            print(msg)

                    elif event.key == K_j or event.key == K_DOWN:
                        self.pos[1] += dpos
                    elif event.key == K_k or event.key == K_UP:
                        self.pos[1] -= dpos
                    elif event.key == K_l or event.key == K_LEFT:
                        self.pos[0] -= dpos
                    elif event.key == K_h or event.key == K_RIGHT:
                        self.pos[0] += dpos
                    elif event.key == K_KP_PLUS:
                        self.angle += 1
                        self.angle = self.angle%361
                    elif event.key == K_KP_MINUS:
                        self.angle -= 1
                        self.angle = self.angle%360
                    elif event.key == K_PAGEUP:
                        dpos += 1
                    elif event.key == K_PAGEDOWN:
                        dpos -= 1

                    elif event.key ==  K_F5:
                        self.marker_scale -= 0.1
                        if self.marker_scale < 0:
                           self.marker_scale = 0.0
                    elif event.key ==  K_F6:
                        self.marker_scale += 0.1

                    elif event.key ==  K_F7:
                        self.camera_scale -= 0.1
                        if self.camera_scale < 0:
                            self.camera_scale = 0.0
                    elif event.key ==  K_F8:
                        self.camera_scale += 0.1

                    elif event.key == K_F9:
                        self.shape_scale -= 10
                        if self.shape_scale < 0:
                            self.shape_scale = 0
                    elif event.key == K_F10:
                        self.shape_scale += 10
                    elif event.key == K_1:
                        self.togle_marker = not self.togle_marker
                    elif event.key == K_2:
                        self.togle_camera = not self.togle_camera
                    elif event.key == K_PRINT:
                        self.print_screen()

                    self.update()
                    pygame.display.update()

    def print_screen(self):
        self.savePrint = False
        now = datetime.now()
        dt_string = now.strftime("%y%m%d-%H%M%S")
        rand_nr = random.randint(1000,9999)
        p = f"{self.savedir}/{dt_string}-{rand_nr}.png"
        # time.sleep(1)
        pygame.image.save_extended(self.display, p)
        print(f"save {p}")
        pass


    def exit(self):
        print("OBS IMPLEMENT THIS TO STORE DATA")
        pass

    def connect_data(self,pdata:posdata):
        self.pdata = pdata

    def overlay_marker(self, id):
        # overlay:Image = self.imgs.aruco_imgs[str(id)]
        # shape = overlay.size
        # ms = self.marker_scale
        # overlay.resize((int(ms*shape[0]),int(ms*shape[1])))
        pos = self.res.loc[str(id)].to_list()
        # self.bg.paste(overlay,list(map(int, pos)),mask=overlay)
        self.cross((pos[0],pos[1]),(0,100,0),label=f"A[{str(id)}]")

    def overlay_camera(self, id):
        pos = self.res.loc[str(id)].to_list()
        self.cross((pos[0],pos[1]),(100,0,0),label=f"C[{str(id)}]")
        pass

    def show_bg(self):
        self.bg = self.imgs.bg_img.copy()
        mode = self.bg.mode
        size = self.bg.size
        data = self.bg.tobytes()
        frame = pygame.image.fromstring(data,size,mode)
        self.display.blit(frame, (0,0))

    def marker_ovrelay(self):
        pos = self.res.loc[str(0)].to_list()
        self.cross(pos,(150,0,0))

    def cross(self, pos, color, label=""):
        length = self.cross_length*2*(np.arctan(self.marker_scale)+1.5)
        width = self.cross_width*2*(np.arctan(self.marker_scale)+1.5)
        vline = [[pos[0],pos[1]+length]
                ,[pos[0],pos[1]-length]]
        hline  = [[pos[0]+length, pos[1]]
                ,[pos[0]-length, pos[1]]]

        pygame.draw.line(self.display, color, vline[0],vline[1],int(width))
        pygame.draw.line(self.display, color, hline[0],hline[1],int(width))
        # Draw text [A/C][posx,posy]
        textsurface = self.myfont.render(f"{label}{list(map(lambda x: int(x), pos))}", False, color)
        self.display.blit(textsurface,(pos[0]+10,pos[1]+5))


    def update(self):
        # self.bg:Image = self.imgs.bg_img.copy()
        self.bg:Image = self.imgs.bg_img
        self.bg = self.bg.resize(list(map(lambda x: int(x*self.window_scale), self.bg.size)))
        cam = self.imgs.camera_img
        # aruco:dict = self.imgs.aruco_imgs
        print(self.pdata)

        if self.pdata is None:
            raise ValueError("Data not connected")

        self.res = self.pdata.transform2D(
                scale = self.shape_scale,
                a     = self.angle,
                dx    = self.pos[0],
                dy    = self.pos[1])
        mode = self.bg.mode
        size = self.bg.size
        data = self.bg.tobytes()
        frame = pygame.image.fromstring(data,size,mode)
        self.display.blit(frame, (0,0))

        for index,content in self.res.iterrows():
            if index.isdigit() and self.togle_marker:
                self.overlay_marker(index)
            elif (not index.isdigit()) and (self.togle_camera):
                self.overlay_camera(index)





def main():
    p = posdata('../results/posdata.csv')
    # print(p.data)
    # print(p.transform2D(100, 90,10,0))
    i = imgs("../report/figures/camera_posese.png")
    i.auto_aruco("../aruco/aruco_{}.png",p)
    i.add_camera("../aruco/camera_subst.png")

    # pygame start
    w = window(i)
    w.connect_data(p)
    w.start()


if __name__ == '__main__':
    main()

