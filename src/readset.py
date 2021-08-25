
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
import matplotlib.pyplot as plt
# import matplotlib as mpl
# import matplotlib.image as mplimg
import pylab
import re
import seaborn as sns
import scipy
# from scipy.stats import f as f_test
from scipy import stats
from matplotlib.cbook import get_sample_data
from typing import Type
from glob import glob
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import multivariate_normal
from itertools import combinations

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
from bioinfokit.analys import stat as binstat
from bioinfokit import analys


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
    indata:dict
    """Stores the images"""
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

        # Pattern matcher for file types used in _save_df
        self.pattern_di = {}
        self.pattern_di['latex'] = re.compile(r"([a-zA-Z0-9\s_\\.\-\(\):])+(.latex|.tex)$")
        self.pattern_di['html'] = re.compile(r"([a-zA-Z0-9\s_\\.\-\(\):])+(.HTML|.htm|.html)$")
        self.pattern_di['png'] = re.compile(r"([a-zA-Z0-9\s_\\.\-\(\):])+(.png|.PNG)$")
        self.pattern_di['csv'] = re.compile(r"([a-zA-Z0-9\s_\\.\-\(\):])+(.csv|.CSV)$")

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
                    #print(trans)
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
        # breakpoint()
        # print(self.indata)
        log.info(self.indata)

    def select_data(self, andarg:dict, request_col=[], df=None):
        """
        Does a SQL like selection based on the dict argumnt
        Supose
        andarg={'filename':'092614.jpg','User','User1'}
        This is equal to:
            SELECT * FROM data
            WHERE filename == 092614.jpg AND User == User1

        The argument request_col select two columns form the set
        df[['filename', 'user']]

        :param dict andarg: Dict that selects rows in df
        :param list request_col: List for selecting columns
        :param pd.DataFrame df: can be used to select data from other df
        """
        if df is None:
            df=self.indata
        if 'label' in andarg.keys():
            andarg['label'] =  andarg['label'].capitalize()
        ret = df.loc[np.all(df[list(andarg)] == pd.Series(andarg), axis=1)]
        log.info(f"Length of data {len(ret)}")
        if request_col and not len(ret) == 0:
            return ret[request_col]
        elif not request_col and not len(ret) == 0:
            return ret
        else:
            # users = df['user'].unique()
            # log.warning(f"Length of output was low {len(ret)} input: {andarg}, users: {users}")
            # breakpoint()
            raise ValueError("Length of output was to low")


    def analyse_feature(self, andarg, rest_col=[]):
        try:
            dd = self.select_data(andarg, rest_col)
            med:pd.Series    = dd.median()
            std:pd.Series    = dd.std()
            cov:pd.DataFrame = dd.cov()
            n:int            = dd.shape[0]
            log.info(f"median:\n{med}\nstd:\n{std}\ncovariance:\n{cov}")
            return med,std,cov,n
        except ValueError as e:
            raise e

    def save_df(self,fname,df:pd.DataFrame):
        p = Path(f"../results/{fname}")
        if   re.fullmatch(self.pattern_di['latex'], fname):
            df.to_latex(p)
        elif re.fullmatch(self.pattern_di['csv'], fname):
            df.to_csv(p)
        elif re.fullmatch(self.pattern_di['html'], fname):
            df.to_html(p)
        elif re.fullmatch(self.pattern_di['png'], fname):
            pass



    def error_calulation(self,  frac=0.5):
        """
        Runs a statistical analysis based on where the picture was taken.
        :param float frac: Divider of human data
        """
        naming=self._conf['ftest']['naming']

        dh = self.select_data({'user':"Human"}, ['user','filename','label', 'u','v'])

        # breakpoint()
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
            if f == "filename":
                print("Remove first line from openpose csv data file")
                raise(KeyError)
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
        col_lab = labels.copy()
        col_lab.append("Direction")
        openp_error_df= pd.DataFrame(columns=col_lab, index=fnames)
        human_error_df= pd.DataFrame(columns=col_lab, index=fnames)
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
        counter_ok = [0,0,0]
        counter_fail = [0,0,0]
        for file in fnames:
            for label in labels:
                # breakpoint()
                log.info(f"file={file}, label={label}")

                # getsub can return a ValueError there fore
                # A test is preformed on each acces.
                try:
                    mean_df:pd.DataFrame = getsub(dh_human)
                    log.info(f"mean_df = \n{mean_df}")
                    counter_ok[0] += 1
                except ValueError as e:
                    counter_fail[0] += 1
                    log.warning(f"Median human: {e}")

                try:
                    sampl_df:pd.DataFrame = getsub(dh_sampl)
                    log.info(f"samp_df = \n{sampl_df}")
                    counter_ok[1] += 1
                except ValueError as e:
                    counter_fail[1] += 1
                    log.warning(f"Sample: {e}")

                try:
                    openp_df:pd.DataFrame = getsub(do_openp)
                    counter_ok[2] += 1
                except ValueError as e:
                    counter_fail[2] += 1
                    log.warning(f"OpenPose: {e}")

                result_list = list()

                if label in error_degdf_df.index:
                    tdef_df = error_degdf_df.loc[label]
                else:
                    tdef_df = [1,0]

                # If mean_df was successfully loaded
                if 'mean_df' in locals():
                    result_list.append(mean_df.shape[0]   + tdef_df[0])
                    thuman_mean = mean_df.mean()
                    mux,muy = thuman_mean.values.tolist()
                    # If openp_df was successfully loaded
                    try:
                        result_list.append(openp_df.shape[0] + tdef_df[1])
                        topenp_mean = openp_df.mean()
                        tx,ty = topenp_mean.values.tolist() # This is NAN?
                        openp_error_df.at[file,label] = terr(mux,muy,tx,ty)
                    except Exception as e:
                        result_list.append(np.nan)
                        # topenp_mean = pd.Series({'u':np.nan, 'v':np.nan})

                    try:
                        error_degdf_df.loc[label] = result_list
                    except ValueError as e:
                        breakpoint()
                        raise e
                    # Human sample
                    if 'sampl_df' in locals():
                        thsamp_mean = sampl_df.mean()
                        tx,ty = thsamp_mean.values.tolist()
                        human_error_df.at[file,label] = terr(mux,muy,tx,ty)
                    # else:
                    #     thsamp_mean = pd.Series({'u':np.nan, 'v':np.nan})

                # breakpoint()
                try:
                    openp_error_df.at[file,"Direction"] = imgloc[file]
                    human_error_df.at[file,"Direction"] = imgloc[file]
                except KeyError as e:
                    print(f"file={file}")
                    breakpoint()


        log.info(f"Human success = {counter_ok[0]/(counter_ok[0]+counter_fail[0])}")
        log.info(f"Sample success = {counter_ok[1]/(counter_ok[1]+counter_fail[1])}")
        log.info(f"OpenPose success = {counter_ok[2]/(counter_ok[2]+counter_fail[2])}")
        errordict = dict()
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
        self.save_df("error_df.latex", errordf)

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
        self.save_df("direction_error_mean.latex", direction_error)
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
        DEPRECATED
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
                "Rshoulder"
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

        ftest_pds = pd.DataFrame(columns=[naming['Fv'], naming['Fr'], naming['Fa'],'s',naming['Tr'],naming['Ta']],index=self.unique['columns'],dtype=np.float32)
        ftest_pds[naming['Fa']] = ftest_pds[naming['Fa']].astype(str)
        ftest_pds[naming['Ta']] = ftest_pds[naming['Ta']].astype(str)
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
                ftest_pds.at[row,naming['Fv']] = f
                p = 1-stats.f.cdf(f, human_deg-1,openp_deg-1)
                log.info(f"p row={row}, f={f}, p={p},{'Reject' if p < 0.5 else 'Accepted'} ")
                ftest_pds.at[row,naming['Fr']] = p
                if not np.isnan(p):
                    ftest_pds.at[row,naming['Fa']] = f"{'Reject' if p < 0.5 else 'Accepted'}"


            except KeyError as e:
                pass
                # breakpoint()
                # raise e
            # T test
        # Needs to be a foor loop to create each column t-test
        for col in self.unique['columns']:
            # c[~(numpy.isnan(c))]
            c1 = self.human_error_df[col].to_numpy(dtype=np.float32)
            c2 = self.openp_error_df[col].to_numpy(dtype=np.float32)
            a=  c1[~(np.isnan(c1))]
            b=  c2[~(np.isnan(c2))]
            if len(a) > 1 and len(b) > 1:
                temp = stats.ttest_ind(
                        a=  a,
                        b=  b,
                        equal_var=True)
                pval = temp.pvalue
                stat = temp.statistic
                ftest_pds.at[col,'s']=stat
                ftest_pds.at[col,naming['Tr']]=pval
                ftest_pds.at[col,naming['Ta']]= f"{'Accept' if pval <= 0.05 else 'Reject'}"



        ftest_pds.style.format("{:.2%}")
        print(ftest_pds)
        self.save_df("ftest_pds.latex", ftest_pds)
        directions = set(self.unique['imgloc'].values())
        ftest_pos_df = pd.DataFrame(
                index=directions,
                columns=[naming['Fv'],naming['Fr'],naming['Fa'],'s',naming['Tr'],naming['Ta']])
        ftest_pos_df[naming['Ta']] = ftest_pos_df[naming['Ta']].astype(str)
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
            ftest_pos_df.at[row,naming['Fv']] = f
            p = 1-stats.f.cdf(f, human_deg-1,openp_deg-1)
            ftest_pos_df.at[row,naming['Fr']] = p
            if not np.isnan(p):
                ftest_pos_df.at[row,naming['Fa']] = f"{'Reject' if p < 0.5 else 'Accepted'}"
            # T -test for direction error:
            human_direction = self.select_data({'Direction':row}, self.unique['columns'], df=self.human_error_df)
            openp_direction = self.select_data({'Direction':row}, self.unique['columns'], df=self.openp_error_df)
            c1 = human_direction.to_numpy(dtype=np.float32)
            c2 = openp_direction.to_numpy(dtype=np.float32)
            a=  c1[~(np.isnan(c1))]
            b=  c2[~(np.isnan(c2))]
            if len(a) > 1 and len(b) > 1:
                temp = stats.ttest_ind(
                        a=  a,
                        b=  b,
                        equal_var=True)
                pval = temp.pvalue
                stat = temp.statistic
                ftest_pos_df.at[row,'s']=stat
                ftest_pos_df.at[row,naming['Tr']]=pval
                ftest_pos_df.at[row,naming['Ta']]= f"{'Accept' if pval <= 0.05 else 'Reject'}"


        print(ftest_pos_df)
        self.ftest_pos_df = ftest_pos_df
        self.save_df("ftest_pos_df.latex", ftest_pos_df)

    def label_selftest(self):
        naming=self._conf['ftest']['naming']
        # data_df = self.indat[['label','u','v','user']]
        human_err_df = self.human_error_df.copy()
        openp_err_df = self.openp_error_df.copy()
        human_err_df["labeller"] = ["human" for x in human_err_df.index]
        openp_err_df["labeller"] = ["opnep" for x in openp_err_df.index]
        data_df = human_err_df.append(openp_err_df)
        # terr = terr[[*self.unique['columns'], "labeller"]]
        cols = self.unique['columns'] # Remove direction from data.
        data_df = data_df[[*cols, "labeller"]]
        data_df = data_df.set_index("labeller")
        # data_df = data_df.T
        melt_df = pd.melt(data_df.reset_index(), id_vars=['labeller'], value_vars=cols)
        melt_df['value'] = melt_df['value'].apply(pd.to_numeric)
        melt_df.columns=["labeller","label","value"]

        res = binstat()
        breakpoint()
        res.tukey_hsd(melt_df,res_var='value', xfac_var=['labeller','label'], anova_model='value ~ C(label)')
        summary_df:pd.DataFrame = res.tukey_summary
        print(summary_df)



    def direction_selftest(self):
        """Directional self test attempts to test if median of south, west, north and east is equal.

        """
        # breakpoint()
        conf = self._conf['directional_selftest']
        naming=self._conf['ftest']['naming']
        acc = conf['accuracy']

        # populate data frame.
        melt_df = pd.melt(self.openp_error_df, id_vars=['Direction'], value_vars=self.unique['columns'])
        melt_df['value'] = melt_df['value'].apply(pd.to_numeric)

        # Binstat is doing the statistics.
        res = binstat()
        res.tukey_hsd(melt_df,res_var='value', xfac_var='Direction', anova_model='value ~ C(Direction)')
        summary_df:pd.DataFrame = res.tukey_summary

        # Store the results in a LaTeX table
        with open("../results/direction_error_df.latex","w") as fp:
            fp.write(summary_df.to_latex())

        # Box plots
        box_plot_conf = conf['box_plot']
        if box_plot_conf['save'] or box_plot_conf['show']:
            x = sns.boxplot(   x='Direction', y='value',    data=melt_df, color='#99c2a2')
            ax = sns.swarmplot(x="Direction", y="value",    data=melt_df, color='#7d0013')
            if box_plot_conf['save']:
                plt.savefig("../results/ftest_againstself_boxplot.pdf")
            if box_plot_conf['show']:
                plt.show()

        hist_plot_conf = conf['hist_plot']
        if hist_plot_conf['save'] or hist_plot_conf['show']:
            fig, axes = plt.subplots(2, 2)
            fig.suptitle('1 row x 2 columns axes with no data')
            subindex=0
            # breakpoint()
            #
            # breakpoint()
            for d in melt_df['Direction'].unique():
                are = hist_plot_conf['arrangement'][d]
                # sns.histplot(melt_df[melt_df == d], x='value',  kde=hist_plot_conf['kde'], ax=axes[are[0],are[1]])
                dd = melt_df[melt_df['Direction'] == d]
                # breakpoint()
                sns.histplot(dd , x='value',  kde=hist_plot_conf['kde'], ax=axes[are[0],are[1]])
                axes[are[0],are[1]].set_title(f"{d}")
            if hist_plot_conf['save']:
                plt.savefig("../results/direction_hist_plot.pdf")
            if hist_plot_conf['show']:
                plt.show()


        # T-test for the melt_df dataframe.






        # res = binstat()
        # res.tukey_hsd(df=dirself,res_var="value",xfac_var='treatments', anova_model='value ~ C(treatments)')
        # # res.tukey_hsd(df=dirself, anova_model='value ~ C(treatments)')
        # print(res.tukey_summary)
        # print(dirself)
        # print(dirself_deg_se)
        # breakpoint()
        # dirlist = list(directions)
        # fval, pval = stats.f_oneway(
        #         dirself[dirlist[0]],
        #         dirself[dirlist[1]],
        #         dirself[dirlist[2]],
        #         dirself[dirlist[3]],
        #         )
        # print(dirself.std())
        # print(dirself.mean())
        # print(dirself.shape)
        # print(f"f value={fval}, p value={pval}")
        # if conf['save_boxplot'] or conf['show_boxplot']:
        #     print("Make box plot")
        #     ax = sns.boxplot(   data=dirself, color='#99c2a2')
        #     ax = sns.swarmplot( data=dirself, color='#7d0013')
        #     if conf['show_boxplot']:
        #         plt.show()
        #     if conf['save_boxplot']:
        #         plt.savefig("../results/ftest_againstself_boxplot.pdf", format="pdf")
        # critical value of f
        # dfn = number of groups - 1
        # dfd = number of labels - number of groups
        # dfn = dirself.shape[1]-1  # wrong shape it should be the shape of the data
        # dfd = dirself.shape[0]-dirself.shape[1]
        #
        # dfn = .shape[1]-1
        # dfd = dirself.shape[0]-dirself.shape[1]
        # crit = scipy.stats.f.ppf(q=1-acc, dfn=dfn, dfd=dfd)
        # cdf  = scipy.stats.f.cdf(crit,     dfn=dfn, dfd=dfd)
        # print(f"Probability density={crit}, Cumulative distribution={cdf}")
        # comb_fn = lambda x: f"{x[0]}-{x[1]}"
        # comb_li = list(map(comb_fn, combinations(directions,2)))
        # comb_raw = combinations(directions,2)
        # comb_df = pd.DataFrame(columns=[naming['Fv'],naming['Pv'],naming['crit'],naming['pp'],naming['cp']],index=comb_li,dtype=float)
        # breakpoint()
        # for comb_it in comb_raw:
        #     print(comb_it)
        #     co = comb_fn(comb_it)
        #     col0 = dirself[comb_it[0]]
        #     col1 = dirself[comb_it[1]]
        #     fval, pval = stats.f_oneway(
        #             col0,col1
        #             )
        #     comb_df.loc[co,naming['Fv']]  = fval
        #     comb_df.loc[co,naming['Pv']]  = pval
        #     crit = scipy.stats.f.ppf(q=1-acc, dfn=dfn, dfd=dfd)
        #     cdf  = scipy.stats.f.cdf(crit,     dfn=dfn, dfd=dfd)
        #     # print(f"f val = {fval} p val = {pval}")

        # # print(combination_df)













    def plot3dstats(self, andarg,  alpha=0.0, outputfp="../results/combo_plot_mlab.png"):
        """
        plot3d plots the stastics for a given data frame.

        :param pd.DataFrame df: The given data frame to show stats for.
        :param str image: Is the image to plot at the bottom.
        """
        #"""
        conf = self._conf['3dplot']
        try:
            med,std,cov,_ = self.analyse_feature(andarg, ['u','v'])
        except ValueError as e:
            errmsg="To few samples in output"
            log.warn(errmsg)
            print(errmsg)
            return -1
        print(1)
        # First make the image
        df = self.select_data(andarg)
        # fig = plt.Figure(figsize=(15,15))
        fig = mlab.figure(bgcolor=(1,1,1))

        if conf['showimg']:
            if 'filename' in andarg.keys() and len(df) > 0:
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
                log.warn(f"andarg has no filename")
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

            # z = zscaling*rv.pdf(zz)
            # mlab.points3d(sample['u'],sample['v'],z,value)
            pplot:mayavi.modules.glyph.Glyph = mlab.points3d(xt,yt,zt,value,scale_factor=1)



            # Showing or saving the figure
            cview = conf['view']
            mlab.view(
                    azimuth=cview['azimuth'],
                    distance=cview['distance'],
                    roll=cview['roll'])
            # mlab.show()
            # mlab.savefig("../results/combo_plot_mlab.png")
            mlab.savefig(outputfp)
            mlab.close()


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

    def create_latex_img_table(self, xcount):
        xwidth = 1/(xcount)
        # breakpoint()
        imgpath = Path(f"{str(self.setdir)}/{self.setconf['imgdir']}")
        exists = imgpath.exists() and imgpath.is_dir()
        log.info("imgpath {imgpath} exists {exists}")
        images = []
        for imgtype in self.setconf['imgtypes']:
            for img in imgpath.glob(imgtype):
                log.info(f"Reading image {img}")
                # breakpoint()
                img = str(img)
                img = img.replace("../datasets","images/datasets")
                images.append(img)
                #report/images/datasets
        ycount = int(len(images)/xcount-1)
        latex = []
        latex.append("\\begin{figure*}\n")
        index = 0
        for col in np.arange(xcount):
            latex.append("\\begin{minipage}[b]{%0.2f\\linewidth}\n"%(xwidth))
            for row in np.arange(ycount+1):
                latex.append("\includegraphics[width=1\linewidth]{%s}\\vspace{4pt}\n"%(images[index]))
                index+=1

            latex.append("\\end{minipage}\n")
        latex.append("\\end{figure*}\n")
        print(index)
        with open("../results/datasets.latex", "w") as fp:
            for row in latex:
                # fp.writelines(row)
                fp.write(row)




        # \subfigure[Input]{
        # \begin{minipage}[b]{0.23\linewidth}
        # \includegraphics[width=1\linewidth]{a1.jpg}\vspace{4pt}
        # \includegraphics[width=1\linewidth]{a2.jpg}\vspace{4pt}
        # \includegraphics[width=1\linewidth]{a3.jpg}\vspace{4pt}
        # \includegraphics[width=1\linewidth]{a4.jpg}
        # \end{minipage}}




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
    data.error_calulation(0.3)
    data.error_t_test()
    # a = {'user':'Human','label':'Rshoulder'}
    # b = {'user':'OpenPose','label':'Rshoulder'}
    # data.t_test()
    a = {'filename':'093614.jpg','label':'Nose'}
    # b = {'filename':'093614.jpg','label':'Nose'}
    # a = {'filename':'093614.jpg'}
    # # print(data.select_data(a))
    # print("--------------")
    # a = {'filename':'093614.jpg','user':'User1','label':'Nose'}
    # a = {'label':'Nose'}
    # print(data.select_data(a))
    # data.plot3dstats(a)

def test_selection():
    data = dataset('P2')
    data.load_anatations()
    ds = []
    ds.append(len(data.select_data({'label':'Rshoulder'})))
    ds.append(len(data.select_data({'label':'Rshoulder'},['u','v'])))
    a = {'filename':'093614.jpg','label':'Rshoulder'}
    ds.append(len(data.select_data(a)))
    a = {'filename':'093614.jpg','label':'Nose'}
    ds.append(len(data.select_data(a)))
    a = {'filename':'093614.jpg','label':'Leye'}
    ds.append(len(data.select_data(a)))
    try:
        a = {'filename':'093614.jpg','label':'Rfoot'}
        ds.append(len(data.select_data(a)))
    except ValueError as e:
        print("Expected to be 0 so its ok")
        ds.append(0)
    print(ds)



if __name__ == '__main__':
    # test_set("P2")
    test_stats("P2")
    # test_selection()
