import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import glob

def find_histogram(clt):

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar
d=0
list = glob.glob("/home/orange/Desktop/Hari.jpg")
for cv_img in list:
    img = cv2.imread(cv_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    #np.savetxt("%s" % list[d] + ".txt", clt.cluster_centers_)
    #fig1 = plt.gcf()
    plt.axis("off")
    Z = [x for _, x in sorted(zip(hist, clt.cluster_centers_), reverse=True)]
    np.savetxt("%s" % list[d] + ".txt", Z)
    print(Z)
    print(hist)
    print(clt.cluster_centers_)
    plt.imshow(bar)
    plt.show()
    plt.close()
    d += 1
