from readset import dataset
from atlas import Atlas
# from atlasCL.Corner import Corner

# from src.camera import camera

def test_set(name):
    """
        Tests a set in the dataset directory.
        :param name:str is the name of the dataset that is loaded with this function
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

def main():
    pass

if __name__ == '__main__':
    test_set("P2")
