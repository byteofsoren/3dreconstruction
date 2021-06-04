from readset import dataset
from atlas import Atlas
# from atlasCL.Corner import Corner

# from src.camera import camera

def test_set(name):
    """
        Tests a set in the dataset directory.
        :param str name: is the name of the dataset that is loaded with this function
    """
    # conf = None
    # with open('./readset.yaml','r') as f:
    #     conf = yaml.load(f,Loader=yaml.FullLoader)
    print("test")
    datap1 = dataset(name)
    myatlas = Atlas(datap1.setconf)
    datap1.set_atlas(myatlas)
    print(f"Created dataset object {name} {datap1} size = {datap1.count}")
    datap1.create_views()
    print("Done. Loaded all images")
    datap1.show_atlas()
    print("builds the atlas")
    datap1.build_atlas()
    print('Sovle geometry')
    # datap1.geometry_solver()
    datap1.create_latex_img_table(4)
    print("end")

def main():
    pass

def input_analysis(name:str,test:list):
    """
        This function calulaces the input mean/variance
        from both the provided human input and the
        open pose input.

        :param str name: Is the name of the set in dataset diretory.
        :param list test: Test method if several is avaible.
    """
    dataobj = dataset(name)
    atlasobj = Atlas(dataobj.setconf)
    dataobj.set_atlas(atlasobj)
    print(f"Created dataset object {name} {dataobj} size = {dataobj.count}")
    dataobj.create_views()




if __name__ == '__main__':
    test_set("P2")
