import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import numpy as np
import glob
import os

class DominantColors:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):
        # read image
        img = cv2.imread(self.IMAGE)

        # convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # save image after operations
        self.IMAGE = img

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_

        # returning after converting to integer from float
        return self.COLORS

d=0
list = glob.glob("/home/orange/Desktop/test/*.jpg")
for cv_img in list:
    clusters = 3
    dc = DominantColors(cv_img, clusters)
    colors = dc.dominantColors()
    # dc.plotHistogram()
    np.savetxt("%s" % list[d] + ".txt", colors)
    d += 1
