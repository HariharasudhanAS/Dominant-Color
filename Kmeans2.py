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
list = glob.glob("/home/orange/Desktop/shape/*.jpg")
for imagesrc in list:
    img = cv2.imread(imagesrc)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)
    cv_img = cv2.imread(imagesrc)
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    Z = np.array([x for _, x in sorted(zip(hist, clt.cluster_centers_), reverse=True)])
    lower_bg = np.array(Z[0,:] - [100,100,100])
    upper_bg = np.array(Z[0,:] + [100,100,100])
    targetbgmask = cv2.inRange(cv_img, lower_bg, upper_bg)
    targetbg = cv2.bitwise_and(cv_img, cv_img, mask=targetbgmask)
    cv2.imwrite("%s" % list[d] + "bg.png", targetbg)
    cv2.imwrite("%s" % list[d] + "bgmask.png", targetbgmask)
    cv_img = cv2.imread(imagesrc)
    lower_fg = np.array(Z[1, :] - [100,100,100])
    upper_fg = np.array(Z[1, :] + [100,100,100])
    targetfgmask = cv2.inRange(cv_img, lower_fg, upper_fg)
    targetfg = cv2.bitwise_and(cv_img, cv_img, mask=targetfgmask)
    cv2.imwrite("%s" % list[d] + "fg.png", targetfg)
    cv2.imwrite("%s" % list[d] + "fgmask.png", targetfgmask)
    cv_img = cv2.imread(imagesrc)
    lower_letter = np.array(Z[2, :] - [100,100,100])
    upper_letter = np.array(Z[2, :] + [100,100,100])
    targetlettermask = cv2.inRange(cv_img, lower_letter, upper_letter)
    targetletter = cv2.bitwise_and(cv_img, cv_img, mask=targetlettermask)
    cv2.imwrite("%s" % list[d] + "letter.png", targetletter)
    cv2.imwrite("%s" % list[d] + "lettermask.png", targetlettermask)
    np.savetxt("%s" % list[d] + ".txt", Z)
   # fig1 = plt.gcf()
    plt.axis("off")
    fig = plt.imshow(bar)
    #fig = cv2.cvtColor(fig, cv2.COLOR_HSV2BGR)
    plt.savefig("%s" % list[d] + "plot.png", bbox_inches='tight')
    plt.close()
    d += 1
