'''
Created on Jul 19, 2016

@author: svensson
'''

import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import os
import glob
import numpy
import lucid2
import unittest
import scipy.misc
from scipy import ndimage

class Test(unittest.TestCase):


    def test_lucid2(self):
        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id30a1/snapshots/snapshots_20160718-152813_Gow8z5"
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id23eh2/snapshots/2070704"
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id23eh2/snapshots/20170822"
        rotation = -90
        for filePath in glob.glob(os.path.join(directory, "*_???.png")):
            fileName = os.path.basename(filePath)
            fileBase = fileName.split(".")[0]
            print(filePath)
            image = scipy.misc.imread(filePath, flatten=True)
            # image = scipy.misc.imrotate(image, -90)
            # im = plt.imread(filePath)
            # im = ndimage.rotate(im, -90)
            imgshape = image.shape
            extent = (0, imgshape[1], 0, imgshape[0])
            implot = plt.imshow(image, extent=extent)
            plt.title(fileBase)
            result = lucid2.find_loop(filePath, opencv2=False)  # , rotation=rotation)
            print(result)
            if result[0] == 'Coord':
                xPos = result[1]
                yPos = imgshape[0] - result[2]
                plt.plot(xPos, yPos, marker='+', markeredgewidth=2,
                         markersize=20, color='black')
            # newFileName = os.path.join(os.path.dirname(filePath), fileBase + "_marked.png")
            # print "Saving image to " + newFileName
            # plt.savefig(newFileName)
            plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
