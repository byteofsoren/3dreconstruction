
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
# from scipy.stats import f as f_test
from scipy import stats
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


# Pandas display formatting.
pd.options.display.float_format='{:.3f}'.format


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
        """
        This function loads the annotated data from the multiple csv files.
        In each set there is a 'set.yaml' containing the sub dict 'annotations'.
        That sub dict contains the following structure.
          - user:
            path:
            column_names:
            delimiter:
            type:
        The user specifies what kind of user produced the data.
        Possible entry's is "Human", "OpenPose" and "Test"
        Then when this function ends it combines the annotations from multiple
        CSV files in to one with an extra column that contains the for mentioned user.

        ---------------------------------------------------------
        | n |  user | label | u | v | filename | width | height |
        ---------------------------------------------------------
        |.

        """
        # The following labda function is used for translating
        # languages from one language to a other
        # a  = lambda inlabel, outlabel: csvfile.loc[(df.label == inlabel),'label']= outlabel
        setdir:str = self.setdir
        atconf = self.setconf['annotations']
        bc = lambda x: f"{bcolors.INF}{x}{bcolors.END}"
        log.info(f"length = {bc(len(atconf))}")
        data = {}
        for csvset in atconf:
            csvpath = Path(f"{setdir}/{csvset['path']}")
            csvname = csvset['user'] # Use this to identify user or openpose
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
                if not csvtype == 0:
                    trans:dict = self._conf['set_translation'][csvtype]
                    print(trans)
                    for lkey in trans.keys():
                        csvfile.loc[(csvfile.label == lkey),'label'] = trans[lkey]
                data[str(csvset['path'])] = csvfile
                data[str(csvset['path'])]['label'] = data[str(csvset['path'])]['label'].str.title()
                data[str(csvset['path'])]['user'] = str(csvname)
            else:
                t = f"file: {str(csvpath)} does {bcolors.WARN}NOT exist{bcolors.END}."
                log.info(t)
                raise ValueError(t)
        # Create total indata table using pandas concat function.
        self.indata = pd.concat([data[key] for key in data.keys()], ignore_index=True)
        # print(self.indata)
        log.info(self.indata)
        # breakpoint()

    def select_data(self, andarg:dict, rest_col=[], df=None):
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
        :param pd.DataFrame df: can be used to select data from other df
        """
        if df is None:
            df=self.indata
        ret = df.loc[np.all(df[list(andarg)] == pd.Series(andarg), axis=1)]
        if not len(rest_col) > 0:
            return ret
        else:
            return ret[rest_col]


    def analyse_feature(self, andarg, rest_col=[]):
        breakpoint()
        dd = self.select_data(andarg, rest_col)
        med:pd.Series    = dd.median()
        std:pd.Series    = dd.std()
        cov:pd.DataFrame = dd.cov()
        n:int            = dd.shape[0]
        log.info(f"median:\n{med}\nstd:\n{std}\ncovariance:\n{cov}")
        return med,std,cov,n

    def error_calulation(self,  frac=0.5):
        """
        Runs a statistical analysis based on where the picture was taken.
        :param float frac: Divider of human data
        """
        naming=self._conf['ftest']['naming']

        dh = self.select_data({'user':"Human"}, ['user','filename','label', 'u','v'])

        do_openp = self.select_data({'user':"OpenPose"}, ['user', 'filename','label', 'u','v'])
        # breakpoint()
        labels:list = dh['label'].unique()
        fnames:list = dh['filename'].unique()
        if isinstance(labels,np.ndarray):
            labels = labels.tolist()
        if isinstance(fnames,np.ndarray):
            fnames = fnames.tolist()
        for l in do_openp['label'].unique():
            if not l in labels:
                labels.append(l)
                log.info(f"label {l} not in {bcolors.SIGN.FIXED}")
            else:
                log.info(f"label {l} was in {bcolors.SIGN.OK} ")
        for f in do_openp['filename'].unique():
            if not f in fnames:
                fnames.append(f)
                log.info(f"filename {f} not in {bcolors.SIGN.FIXED}")
            else:
                log.info(f"filename {f} was in {bcolors.SIGN.OK} ")

        # Random divider of human data.
        # Total error calculation.
        # Total f test
        # Total t test
        # error calculation for each South, west, north and east
        # f test for each South, West, North and East
        # t test for each South, West, North and East
        #
        # 1. Random divider of human data
        dh_sampl:pd.DataFrame = dh.sample(frac=(1-frac))
        dh_human:pd.DataFrame = dh.drop(dh_sampl.index)
        # dh_human:pd.DataFrame = dh.sample(frac=(1-frac))
        # dh_sampl:pd.DataFrame = dh.drop(dh_human.index)


        # Build median mu matrix for each label in each image
        # The mumtx is the error matrix with the shape:
        #           foot | head | arm |
        # 2343.jpg
        # 6543.jpg
        openp_error_df= pd.DataFrame(columns=labels, index=fnames)
        human_error_df= pd.DataFrame(columns=labels, index=fnames)
        # breakpoint()
        error_degdf_df = pd.DataFrame(columns=[naming['hd'],naming['od']])

        self.unique = {'columns':labels, 'index':fnames}
        # How to update test.at['2134.jpg','foot']= 34
        terr = lambda mux,muy,tx,ty : np.sqrt((mux-tx)**2 + (muy-ty)**2)
        getsub = lambda df: self.select_data({'filename':file,'label':label},['u','v'],df=df)
        imgloc = self.setconf['imgloc']
        self.unique['imgloc'] = imgloc
        # imgloc stores the approximated positon of where the camera is located.
        # imgloc['2543.jpg'] -> South means that the picture was taken from
        # the feats wile North means form the head.
        for file in fnames:
            for label in labels:
                log.info(f"file={file}, label={label}")
                mean_df:pd.DataFrame = getsub(dh_human)
                # log.info(f"mean_df shape = {mean_df.shape}")
                log.info(f"mean_df = \n{mean_df}")
                sampl_df:pd.DataFrame = getsub(dh_sampl)
                # log.info(f"samp_df shape = {sampl_df.shape}")
                log.info(f"samp_df = \n{sampl_df}")
                # breakpoint()
                openp_df:pd.DataFrame = getsub(do_openp)
                thuman_mean = mean_df.mean()
                # log.warn(f"thuman_mean shape = {thuman_mean.shape}")
                # Store the degres of freedom

                tdefdf = list()
                if label in error_degdf_df.index:
                    tdef_df = error_degdf_df.loc[label]
                else:
                    tdef_df = [0,0]
                tdefdf.append(mean_df.shape[0]   + tdef_df[0])
                tdefdf.append(openp_df.shape[0] + tdef_df[1])
                error_degdf_df.loc[label] = tdefdf


                thsamp_mean = sampl_df.mean()
                topenp_mean = openp_df.mean()
                # Median calulation
                mux,muy = thuman_mean.values.tolist()

                # Our test cases.
                # Human sample
                tx,ty = thsamp_mean.values.tolist()
                perr=terr(mux,muy,tx,ty)
                human_error_df.at[file,label] = perr
                human_error_df.at[file,"Direction"] = imgloc [file]
                # log.info(f"OK human error {perr:.2f}")
                # OpenPose sample
                tx,ty = topenp_mean.values.tolist()
                perr=terr(mux,muy,tx,ty)
                openp_error_df.at[file,label] = perr
                openp_error_df.at[file,"Direction"] = imgloc[file]
                # log.info(f"OK OpenPose error {perr:.2f}")

        errordict = dict()
        # breakpoint()
        print("Cumulative error for a human:")
        errordict[naming['hm']] = human_error_df.mean()
        errordict[naming['hv']] = human_error_df.var()
        print("Cumulative error for a OpenPose:")
        errordict[naming['om']] = openp_error_df.mean()
        errordict[naming['ov']] = openp_error_df.var()
        errordf = pd.DataFrame(errordict)

        # Cumdict creates an row for mean
        # and variance to apply to error dict.
        cumdict = dict() # <-- Important data found here:
        cumdict[naming['hm']] = human_error_df.mean().mean()
        cumdict[naming['hv']]  = human_error_df.mean().var()
        cumdict[naming['om']] = openp_error_df.mean().mean()
        cumdict[naming['ov']]  = openp_error_df.mean().var()
        s = pd.Series(cumdict,name='Total mean/variance:')
        self.errordf = errordf.append(s,ignore_index=False)
        print(errordf)

        # directional error calculation.
        # human_direction=dict()
        # openp_direction=dict()
        direction_error= pd.DataFrame(
                columns=[
                    naming['hm'],
                    naming['hv'],
                    naming['om'],
                    naming['ov']
                    ],
                index=set(imgloc.values())) # <-- More important data found here
        direction_degdf_df = pd.DataFrame(
                columns=[
                    naming['hd'],
                    naming['od'],
                    ],
                index=set(imgloc.values()))
        for direction in set(imgloc.values()):
            human_direction = self.select_data({'Direction':direction},df=human_error_df)
            openp_direction = self.select_data({'Direction':direction},df=openp_error_df)

            direction_error.at[direction,naming['hm']] = human_direction.mean().mean()
            direction_error.at[direction,naming['hv']] = human_direction.mean().var()
            direction_error.at[direction,naming['om']] = openp_direction.mean().mean()
            direction_error.at[direction,naming['ov']] = openp_direction.mean().var()


            temp_deg = list()
            temp_deg.append(human_direction.count().sum())
            temp_deg.append(openp_direction.count().sum())
            direction_degdf_df.loc[direction] = temp_deg


        print(direction_error)
        self.direction_error_df:pd.DataFrame = direction_error
        self.error_df:pd.DataFrame  = errordf
        self.error_degdf_df:pd.DataFrame = error_degdf_df
        self.human_error_df = human_error_df
        self.openp_error_df = openp_error_df
        print(error_degdf_df)
        print(direction_degdf_df)
        self.direction_degdf_df = direction_degdf_df


    def error_t_test(self, user1:str='Human',user2='OpenPose'):
        """
        Error t-test does both a f-test and t-test on the given data in
        the pandas frames:
            self.direction_error_df
            self.error_df

        The dict self.unique contains:
            "index":
                "1233.jpg"
                "2332.jpg"
                "1323.jpg"
            "columns"
                "Rsholder"
                "Relbow"...
            "imgloc" ie image location:
                "123.jpg":"South"
                "124.jpg":"West"
                "131,jpg":"North"
                "125.jpg":"East"
        """
        # Naming provides synonym's to as effort to keep symmetry in data.
        naming=self._conf['ftest']['naming']
        # Raw data if needed.
        # raw_human_df= self.select_data({'user':user1}, ['user','filename','label', 'u','v'])

        # raw_openp_df= self.select_data({'user':user2}, ['user','filename','label', 'u','v'])

        ftest_pds = pd.DataFrame(columns=['f', 'p_f', 'f accept','s','p_t','t accept'],index=self.unique['columns'],dtype=np.float32)
        ftest_pds['f accept'] = ftest_pds['f accept'].astype(str)
        # f_test.cdf(sigma,dfn,dfm)
        for row in self.unique['columns']:
            try:
                # F test
                human_var = self.error_df.at[row,naming['hv']]
                openp_var = self.error_df.at[row,naming['ov']]
                human_deg = self.error_degdf_df.at[row,naming['hd']]
                openp_deg = self.error_degdf_df.at[row,naming['od']]
                log.info(row)
                if human_var>= openp_var:
                    f = human_var/openp_var
                else:
                    f = openp_var/human_var
                ftest_pds.at[row,'f'] = f
                p = 1-stats.f.cdf(f, human_deg-1,openp_deg-1)
                log.info(f"p row={row}, f={f}, p={p},{'Reject' if p < 0.5 else 'Accepted'} ")
                ftest_pds.at[row,'p_f'] = p
                if not np.isnan(p):
                    ftest_pds.at[row,'f accept'] = f"{'Reject' if p < 0.5 else 'Accepted'}"



            except KeyError as e:
                breakpoint()
                raise e
        # T test
        breakpoint()
        temp = stats.ttest_ind(
                a=self.human_error_df,
                b=self.openp_error_df,
                equal_var=True)
        ftest_pds.style.format("{:.2%}")
        print(ftest_pds)
        directions = set(self.unique['imgloc'].values())
        ftest_pos_df = pd.DataFrame(
                index=directions,
                columns=['f','p_f','f accept'])
        for row in directions:
            human_mean = self.direction_error_df.at[row,naming['hm']]
            openp_mean = self.direction_error_df.at[row,naming['om']]
            human_var  = self.direction_error_df.at[row,naming['hv']]
            openp_var  = self.direction_error_df.at[row,naming['ov']]
            log.info(f"{row}-> human:{human_mean,human_var} openp:{openp_mean,openp_var}")
            human_deg = self.direction_degdf_df.at[row,naming['hd']]
            openp_deg = self.direction_degdf_df.at[row,naming['od']]

            if human_var>= openp_var:
                f = human_var/openp_var
            else:
                f = openp_var/human_var
            # ftest_pds.at[row,'f'] = f
            ftest_pos_df.at[row,'f'] = f
            p = 1-stats.f.cdf(f, human_deg-1,openp_deg-1)
            ftest_pos_df.at[row,'p_f'] = p
            if not np.isnan(p):
                ftest_pos_df.at[row,'f accept'] = f"{'Reject' if p < 0.5 else 'Accepted'}"
        print(ftest_pos_df)





























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
        fig = mlab.figure(bgcolor=(1,1,1))

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
                # obj.actor.scale = [0.8, 0.8, 0.8]
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
                # breakpoint()
                obj.actor.rotate_z(90)
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

            # rv.pdf is the f(x,y)=z function
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
            # grid.actor.rotate_z(180)

            # Shows the outline box aruond the figure
            mlab.outline(
                    color=(0, 0, 0),
                    opacity=0.8
                    )
            # plot the scatter for collected data:
            print(sample)
            print(sample.shape)
            xt = []
            yt = []
            zt = []
            tt = sample/conf['resize']
            value = [1*conf['markersize'] for _ in np.arange(0,sample.shape[0])]
            gt = lambda i,j: int(tt.iloc[i][j])
            w = np.max(imgsize['width'])
            for i in np.arange(0,sample.shape[0]):
                xt.append(-1*gt(i,0)*conf['resize'] + w)
                yt.append(1*gt(i,1)*conf['resize'])
                zt.append(z[gt(i,0), gt(i,1)])

            # breakpoint()
            # z = zscaling*rv.pdf(zz)
            # mlab.points3d(sample['u'],sample['v'],z,value)
            pplot:mayavi.modules.glyph.Glyph = mlab.points3d(xt,yt,zt,value,scale_factor=1)
            # breakpoint()



            # Showing or saving the figure
            cview = conf['view']
            mlab.view(
                    azimuth=cview['azimuth'],
                    distance=cview['distance'],
                    roll=cview['roll'])
            # mlab.show()
            mlab.savefig("../logs/combo_plot_mlab.png")






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
    # a = {'user':'Human','label':'Rsholder'}
    # b = {'user':'OpenPose','label':'Rsholder'}
    data.error_calulation(0.3)
    data.error_t_test()
    # data.t_test()
    # a = {'filename':'093614.jpg','label':'Nose',' k'
    # b = {'filename':'093614.jpg','label':'Nose'}
    # a = {'filename':'093614.jpg'}
    # # print(data.select_data(a))
    # print("--------------")
    # a = {'filename':'093614.jpg','user':'User1','label':'Nose'}
    # a = {'label':'Nose'}
    # print(data.select_data(a))
    # data.plot3dstats(a)





if __name__ == '__main__':
    # test_set("P2")
    test_stats("P2")
