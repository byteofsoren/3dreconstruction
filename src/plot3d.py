from readset import dataset

def create3d(name):
    # log.info(f"Test {name}")
    data = dataset(name)
    data.load_anatations()
    data.error_calulation(0.3)
    # a = {'user':'Human','label':'Rsholder'}
    # b = {'user':'OpenPose','label':'Rsholder'}
    # data.t_test()
    a = {'filename':'093614.jpg','label':'RSholder','user':'Human'}
    data.plot3dstats(a)
    a = {'filename':'093450.jpg','label':'RSholder','user':'Human'}
    data.plot3dstats(a, outputfp="../results/093450_H_RSholder.png")
    a = {'filename':'093450.jpg','label':'RSholder','user':'OpenPose'}
    data.plot3dstats(a, outputfp="../results/093450_O_RSholder.png")


if __name__ == '__main__':
    create3d("P2")
