import numpy as np
from matplotlib import pyplot as plt
import pydicom
import cv2


file = "/Users/adnanhafeez/Desktop/RI.QC_EPID_U07.MV_0_0b.dcm"
ds = pydicom.read_file(file)


def rescale(input):
    # rescale original 16 bit image to 8 bit values [0,255]
    x0 = input.min()
    x1 = input.max()
    y0 = 0
    y1 = 256.0
    i8 = ((input - x0) * ((y1 - y0) / (x1 - x0))) + y0
    # create new array with rescaled values and unsigned 8 bit data type
    o8 = i8.astype(np.uint8)
    return o8


def Otsu(im):
    [hist, _] = np.histogram(im, bins=256, range=(0, 255))
    # Normalization so we have probabilities-like values (sum=1)
    hist = 1.0*hist/np.sum(hist)

    val_max = -999
    thr = -1
    opti = 45
    for t in range(1,255):
        # Non-efficient implementation
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        m1 = np.sum(np.array([i for i in range(t)])*hist[:t])/q1
        m2 = np.sum(np.array([i for i in range(t,256)])*hist[t:])/q2
        val = q1*(1-q1)*np.power(m1-m2,2)
        if val_max < val:
            val_max = val
            thr = t

    print("Threshold: {}".format(thr))



    x = (im < thr)*1
    width = np.sum(x[int(x.shape[0]/2),])
    height = np.sum(x[:,int(x.shape[1]/2)])
    #print(width, height)
    #print(x)
    #plt.subplot(121)
    #plt.imshow(im)
    #plt.subplot(122)
    #plt.imshow(im > thr)
    #plt.show()
    print("Width = ", width, "Height = ", height, ", Field Edges = ", findFieldEdge(-x.astype(np.uint8)))
    plt.imshow(im)
    #plt.arrow(findFieldEdge(-x.astype(np.uint8))[0], findFieldEdge(-x.astype(np.uint8))[1]*2, width, 0, head_width=25, head_length=40, linewidth=2, color='r', length_includes_head=True,overhang=-0.2)
    #plt.arrow(findFieldEdge(-x.astype(np.uint8))[0]*2, findFieldEdge(-x.astype(np.uint8))[1], 0, height, head_width=25,
    #          head_length=40, linewidth=2, color='r', length_includes_head=True)


    plt.annotate('', horizontalalignment='center',
                xy=(findFieldEdge(-x.astype(np.uint8))[0]*2, findFieldEdge(-x.astype(np.uint8))[1]), xycoords='data',
                xytext=(findFieldEdge(-x.astype(np.uint8))[0]*2, findFieldEdge(-x.astype(np.uint8))[1]+height), textcoords='data',
                arrowprops=dict(arrowstyle="<->",
                                connectionstyle="arc3", edgeColor='g'),
                )

    plt.annotate(height, xy=(findFieldEdge(-x.astype(np.uint8))[0]*2, findFieldEdge(-x.astype(np.uint8))[1]+height/2),
                 xycoords='data', textcoords='data', ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))

    plt.annotate('', horizontalalignment='center',
                 xy=(findFieldEdge(-x.astype(np.uint8))[0], findFieldEdge(-x.astype(np.uint8))[1]*2), xycoords='data',
                 xytext=(findFieldEdge(-x.astype(np.uint8))[0]+width, findFieldEdge(-x.astype(np.uint8))[1]*2),
                 textcoords='data',
                 arrowprops=dict(arrowstyle="<->",
                                 connectionstyle="arc3", edgeColor='g')
                 )
    plt.annotate(width,
                 xy=(findFieldEdge(-x.astype(np.uint8))[0]+width/2, findFieldEdge(-x.astype(np.uint8))[1]*2),
                 xycoords='data', textcoords='data', ha='center',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))

    plt.show()
    print("Field Size:", 100/150*0.0392*width, "x", 100/150*0.0392*height, 'cm')


def findFieldEdge(volume):
    contours, _ = cv2.findContours(volume,0,1)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    #img = cv2.rectangle(volume, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.circle(img, (x, y), 5, (78, 55, -128), 1)
    #print(w,h)
    #print(x,y)
    #cv2.imshow("Radiation Field", img)
    #cv2.waitKey()
    return x, y


Otsu(rescale(ds.pixel_array))
