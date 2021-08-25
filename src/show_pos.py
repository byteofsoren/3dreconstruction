from readset import dataset

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


class TF:
    """docstring for TF"""
    tf_z = lambda self, a,dx,dy,dz: np.array([[C(a),-S(a),0,dx],[S(a),C(a),0,dy],[0,0,1,0],[0,0,0,1]])
    tf_2D = lambda self, a,dx,dy: np.array([[C(a),-S(a),dx],[S(a),C(a),dy],[0,0,1]])
    alpha = lambda self, a: a*(np.pi/180)
    def __init__(self,data, matrix=None, matrix2D=None):
        self.data =     data
        self.matrix =   matrix
        self.matrix2D = matrix2D.astype(int)

    def transform(self,scale,a,dx,dy,dz):
        if self.matirx is None:
            raise ValueError("No matix is connected")
        points:np.ndarray = scale*self.matrix.T
        points =  np.vstack((points,np.ones((points.shape[1]))))
        res =  self.tf_z(0,dx,dy,dz)@self.tf_z(self.alpha(a),0,0,0)@points
        return pd.DataFrame(res[:-1].T, index=self.data.index)

    def transform2D(self,scale,a,dx,dy):
        if not self.matrix2D is None:
            inv = np.array([[1,0],[0,1]])
            try:
                points:np.ndarray = scale*inv@self.matrix2D.T
            except TypeError as e:
                print(f"scale={scale}")
                print(f"inv=\n{inv}")
                print(self.matrix2D)
                breakpoint()
            # points:np.ndarray = scale*inv@self.matrix2D.T
            points =  np.vstack((points,np.ones((points.shape[1]))))
            res =  self.tf_2D(0,dx,dy)@self.tf_2D(self.alpha(a),0,0)@points
            return pd.DataFrame(res[:-1].T, index=self.data.index)
        else:
            raise ValueError("No data provided")


class posdata(TF):
    """docstring for posdata"""
    def __init__(self, filename:str):
        self.fname= Path(filename)
        # Do the file exists?
        if (not self.fname.exists()) or (not self.fname.is_file()):
            raise OSError(filename)
        print("[Ok] file exits")
        # Read the data.
        self.data = pd.read_csv(str(self.fname),index_col=0)
        self.matrix = self.data[['X','Y','Z']].to_numpy()
        self.matrix2D = self.data[['X','Y']].to_numpy(dtype=int)
        super(TF, self).__init__(self.matrix,self.matrix2D)




class features(TF):
    """docstring for features"""
    img_selected = None
    def __init__(self, dataset:dataset):
        self.dataset = dataset
        # print(self.dataset.indata)
        self.human_df:pd.DataFrame = self.dataset.select_data({'user':'Human'},    request_col=['label', 'u','v','filename'])
        self.openp_df:pd.DataFrame = self.dataset.select_data({'user':'OpenPose'}, request_col=['label', 'u','v','filename'])
        self.human_df = self.human_df.set_index('label')
        self.openp_df = self.openp_df.set_index('label')
        self.images = self.dataset.indata['filename'].unique()
        self.img_selected = self.images[0]
        self.matrix2D = None

    def select_id(self, id):
        if  0 <= id and id < len(self.images):
            self.img_selected = self.images[id]
        else:
            print(f"id={id} is out of rage")

    def slect_index(self, index:str ):
        if index[0].lower() == "h":
            self.data = self.dataset.select_data({'filename':self.img_selected},request_col=['u','v'],df=self.human_df)
            self.matrix2D = self.data[['u','v']].to_numpy(dtype=int)
        elif index[0].lower() == "o":
            self.data = self.dataset.select_data({'filename':self.img_selected},request_col=['u','v'],df=self.openp_df)
            self.matrix2D = self.data[['u','v']].to_numpy(dtype=int)




class imgs:
    """docstring for imgs"""
    camera_img:Image
    bg_img:Image # Contain the image
    bg:str       # Contain the image path
    def __init__(self, bgfname:str):
        self.bg = bgfname
        bg = Path(bgfname)
        if (not bg.is_file()) or (not bg.exists()):
            raise OSError(bgfname)
        self.bg_img = Image.open(str(bg))

class imgs_set(imgs):
    """docstring for imgs_set"""

    bg_img:Image # Contain the image
    bg:list      # Contain the image paths

    def __init__(self, dataset:dataset):
        path = dataset.setdir
        self._imgpath = Path(f"{path}/images/")
        self.bg = dataset.indata['filename'].unique()
        self.img_select(0)



    def img_select(self,id):
        if id < 0:
            return 0;
        elif len(self.bg) <= id:
            return len(self.bg)
        else:
            bg = Path(f"{self._imgpath}/{self.bg[id]}")
            self.bg_img = Image.open(str(bg))
            return id







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
    bg_img_count = 0
    dpos = 1
    angle = 0
    lable_select=0


    def __init__(self, imgobj:imgs,window_scale=0.4):
        print("my window")
        self.imgs = imgobj
        self.window_scale = window_scale
        shape = imgobj.bg_img.size
        # Display object is created here
        self.display = pygame.display.set_mode(list(map(lambda x: int(x*self.window_scale), shape)))
        pygame.display.set_caption("Data corrector")

        self.color = (255,255,255)
        self.display.fill(self.color)

        pygame.font.init() # you have to call this at the start,
                   # if you want to use this module.
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self.display_font = pygame.font.SysFont('Comic Sans MS', 30)
        # textsurface = self.myfont.render('Some Text', False, (0, 0, 0))

        # Reder bg
        self.cross_length = int(10*self.window_scale)
        self.cross_width  = int(5*self.window_scale)
        pygame.display.update()



    def start(self):
        run = True
        self.dpos = 10
        self.angle = 0
        show_text = True
        while run:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    show_text = True
                    if event.key == K_ESCAPE:
                        self.exit()
                        pygame.quit()
                        sys.exit()
                    elif event.key == K_SPACE:
                        self.savePrint = True

                    elif event.key == K_j or event.key == K_DOWN:
                        self.pos[1] += self.dpos
                    elif event.key == K_k or event.key == K_UP:
                        self.pos[1] -= self.dpos
                    elif event.key == K_l or event.key == K_LEFT:
                        self.pos[0] -= self.dpos
                    elif event.key == K_h or event.key == K_RIGHT:
                        self.pos[0] += self.dpos
                    elif event.key == K_KP_PLUS:                            # + Angle + 1 deg
                        self.angle += 1
                        self.angle = self.angle%361
                    elif event.key == K_KP_MINUS:                           # - Angle -1 deg
                        self.angle -= 1
                        self.angle = self.angle%360
                    elif event.key == K_PAGEUP:                             # PageUp dx,dy + 1
                        self.dpos += 10
                    elif event.key == K_PAGEDOWN:                           # PageDown dx,dy -1
                        self.dpos -= 10

                    elif event.key == K_F1:
                        # Label type 0 to 5
                        if self.lable_select == 5:
                            self.lable_select = 0
                        else:
                            self.lable_select += 1

                    elif event.key == K_F4:
                        self.bg_img_count += 1
                        self.features.select_id(self.bg_img_count)
                    elif event.key == K_F3:
                        self.bg_img_count -= 1
                        self.features.select_id(self.bg_img_count)

                    elif event.key ==  K_F5:
                        self.marker_scale -= 0.1
                        if self.marker_scale < 0:
                           self.marker_scale = 0.0
                        print(self.marker_scale)
                    elif event.key ==  K_F6:
                        self.marker_scale += 0.1
                        print(self.marker_scale)

                    # elif event.key ==  K_F7:
                    #     self.camera_scale -= 0.1
                    #     if self.camera_scale < 0:
                    #         self.camera_scale = 0.0
                    # elif event.key ==  K_F8:
                    #     self.camera_scale += 0.1

                    elif event.key == K_F9:                         # F9  Scaling +
                        self.shape_scale -= 0.01
                        if self.shape_scale < 0:
                            self.shape_scale = 0
                    elif event.key == K_F10:                        # F10 Scaling -1
                        self.shape_scale += 0.01
                    elif event.key == K_1:
                        self.togle_marker = not self.togle_marker   # 1 Toggle features and markers
                    elif event.key == K_2:
                        self.togle_camera = not self.togle_camera
                    elif event.key == K_PRINT:
                        self.print_screen()

                    self.update()
                pygame.display.update()


    def update(self):
        # Select bg dependent on type
        if type(self.imgs.bg) is str:
            # Only one static bg
            self.bg = self.imgs.bg_img
        if type(self.imgs.bg) is list or type(self.imgs.bg) is np.ndarray:
            # Numerous bgs can be selected rotation with F3 and F4.
            self.bg_img_count = self.imgs.img_select(self.bg_img_count)
            self.bg = self.imgs.bg_img
        # Resize the bg to fit the screen
        self.bg = self.bg.resize(list(map(lambda x: int(x*self.window_scale), self.bg.size)))
        mode = self.bg.mode
        size = self.bg.size
        data = self.bg.tobytes()
        # Create a frame
        self.display.fill(self.color)
        frame = pygame.image.fromstring(data,size,mode)
        # Display the frame
        self.display.blit(frame, (0,0))

        # if corners and cameras are connected in 2D
        if not self.pdata is None:
            try:
                self.features.slect_index("OpenPose")
                self.res = self.pdata.transform2D(
                        scale = self.shape_scale,
                        a     = self.angle,
                        dx    = self.pos[0],
                        dy    = self.pos[1])
                for index,content in self.res.iterrows():
                    if index.isdigit() and self.togle_marker:
                        self.overlay_marker(index)
                    elif (not index.isdigit()) and (self.togle_camera):
                        self.overlay_camera(index)
            except ValueError as e:
                pass

        # If Features are connected in 2D
        if not self.features is None:
            try:
                if self.togle_marker:
                    self.features.slect_index("Human")
                    self.res = self.features.transform2D(
                            scale = self.shape_scale,
                            a     = self.angle,
                            dx    = self.pos[0],
                            dy    = self.pos[1])
                    for index,content in self.res.iterrows():
                        self.overlay_feature(index,"H",color=(255,0,0))
                        # self.overlay_feature(index,"H",color=(255,0,0))
                        pass
            except ValueError as e:
                print("Human has an error")
            try:
                self.features.slect_index("OpenPose")
                self.res = self.features.transform2D(
                        scale = self.shape_scale,
                        a     = self.angle,
                        dx    = self.pos[0],
                        dy    = self.pos[1])
                for index,content in self.res.iterrows():
                    # breakpoint()
                    self.overlay_feature(index,"O",color=(0,255,0))
                    pass
            except ValueError as e:
                print("OpenPose has an error")
            text_displace = 20
            i=0

            text_display = list()
            text_display.append(f"Pos={self.pos}, dpos={self.dpos}, angle={self.angle:.2f} exit by [esc]")
            text_display.append(f"shape_scale={self.shape_scale:.2f}, marker_scale={self.marker_scale:.2f}")
            for msg in text_display:
                textsurface = self.display_font.render(msg, False, (160, 160, 160))
                self.display.blit(textsurface,(0,i))
                i+= text_displace
            if self.savePrint:
                self.print_screen()

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

    def connect_featuers(self, features:features):
        self.features = features

    def overlay_marker(self, id):
        pos = self.res.loc[str(id)].to_list()
        # self.bg.paste(overlay,list(map(int, pos)),mask=overlay)
        self.cross((pos[0],pos[1]),(0,100,0),label=f"A[{str(id)}]")


    def show_bg(self):
        if self.imgs.bg is str:
            self.bg = self.imgs.bg_img.copy()
        if self.imgs.bg is list:
            self.bg_img_count = self.imgs.img_select(self.bg_img_count)
            self.bg = self.imgs.bg_img.copy()
        mode = self.bg.mode
        size = self.bg.size
        data = self.bg.tobytes()
        frame = pygame.image.fromstring(data,size,mode)
        self.display.blit(frame, (0,0))

    def overlay_camera(self, id):
        pos = self.res.loc[str(id)].to_list()
        self.cross((pos[0],pos[1]),(100,0,0),label=f"C[{str(id)}]")
        pass

    def marker_ovrelay(self):
        pos = self.res.loc[str(0)].to_list()
        self.cross(pos,(150,0,0))

    def overlay_feature(self,id,label, color=(127,127,127),df=None):
        if not df is None:
            pos = df.loc[str(id)]
        else:
            pos:pd.DataFrame = self.res.loc[str(id)]

        # What label should be seen?
        slabel = None
        showpos = False
        if self.lable_select == 1:
            slabel = str(id)
            showpos = False
        elif self.lable_select == 2:
            slabel = str(id)
            showpos = True
        elif self.lable_select == 3:
            slabel = str(label)
            showpos = False
        elif self.lable_select == 4:
            slabel = str(label)
            showpos = True
            pass



        shape = None
        if isinstance(pos, pd.DataFrame):
            if pos.shape[0] == 1:
                # Just one feature
                self.cross(pos,color, label=slabel, showpos=showpos)
            elif pos.shape[0] > 0:
                # Loop for all features
                for it,po in pos.iterrows():
                    self.cross(po.to_list(),color, label=slabel, showpos=showpos)
        if isinstance(pos, pd.Series):
                self.cross(pos,color, label=slabel, showpos=showpos)



    def cross(self, pos, color, label="", showpos=False):
        # length = self.cross_length*2*(np.arctan(self.marker_scale)+1.5)
        # width = self.cross_width*2*(np.arctan(self.marker_scale)+1.5)
        length = self.cross_length*(1+self.marker_scale)
        width =  self.cross_width *(1+self.marker_scale)
        vline = [[pos[0],pos[1]+length]
                ,[pos[0],pos[1]-length]]
        hline  = [[pos[0]+length, pos[1]]
                ,[pos[0]-length, pos[1]]]

        pygame.draw.line(self.display, color, vline[0],vline[1],int(width))
        pygame.draw.line(self.display, color, hline[0],hline[1],int(width))
        # Draw text [A/C][posx,posy]
        textsurface = None
        if showpos and label is not None:
            textsurface = self.myfont.render(f"{label}{list(map(lambda x: int(x), pos))}", False, color)
        elif label is not None:
            textsurface = self.myfont.render(f"{label}", False, color)
            pass

        if textsurface is not None:
            self.display.blit(textsurface,(pos[0]+10,pos[1]+5))








def corners():
    p = posdata('../results/posdata.csv')
    # print(p.data)
    # print(p.transform2D(100, 90,10,0))
    i = imgs("../report/figures/camera_posese.png")

    # pygame start
    w = window(i)
    w.connect_data(p)
    w.start()

def data_points():
    """data_points fitts a dataset image to a set of anotations.

    @raise e:  Description
    """
    data = dataset("P2")
    data.load_anatations()
    f = features(data)
    i = imgs_set(data)
    w = window(i,window_scale=0.3)
    w.connect_featuers(f)
    w.angle = -30
    w.pos = [20,20]
    w.shape_scale = 0.29
    w.marker_scale = 3.5
    w.start()
    # print(f.human_np)
    # f.select_id(2)
    # print(f.human_np)


def main():
    # corners()
    data_points()



if __name__ == '__main__':
    main()

