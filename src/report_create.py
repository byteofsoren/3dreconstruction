from readset import dataset
from atlas import Atlas

def report(name):
    """
        Tests a set in the dataset directory.
        :param str name: is the name of the dataset that is loaded with this function
    """
    # conf = None
    # with open('./readset.yaml','r') as f:
    #     conf = yaml.load(f,Loader=yaml.FullLoader)
    print("start")
    data = dataset(name)
    data.load_anatations()
    data.error_calulation(0.3)
    data.error_t_test()
    data.create_latex_img_table(3)
    print("end")


if __name__ == '__main__':
    report("P2")
