from readset import dataset

def create3d(name):
    # log.info(f"Test {name}")
    data = dataset(name)
    data.load_anatations()
    data.error_calulation(0.3)
    # a = {'user':'Human','label':'Rshoulder'}
    # b = {'user':'OpenPose','label':'Rshoulder'}
    # data.t_test()
    a = {'filename':'093614.jpg','label':'Rshoulder','user':'Human'}
    data.plot3dstats(a)
    a = {'filename':'093450.jpg','label':'Rshoulder','user':'Human'}
    data.plot3dstats(a, outputfp="../results/093450_H_Rshoulder.png")
    a = {'filename':'093450.jpg','label':'Rshoulder','user':'OpenPose'}
    data.plot3dstats(a, outputfp="../results/093450_O_Rshoulder.png")


if __name__ == '__main__':
    create3d("P2")
