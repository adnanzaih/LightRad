import argparse
import pydicom
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.morphology import black_tophat


def Analyze(input):
    # perhaps we want to first normalize the image so the maximum pixel is 255, and minimum is 0
    img = pydicom.dcmread(input)
    normalizedImg = rescale(img.pixel_array)
    normalizedImgO = normalizedImg.copy()
    normalizedImg = (black_tophat(normalizedImg, disk(12)))
    normalizedImg = cv2.GaussianBlur(normalizedImg, (5, 5), 3)
    # apply hough transform
    circles = cv2.HoughCircles(normalizedImg, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=10, minRadius=3, maxRadius=6)
    # place circles and cente rectangle on image
    if circles is not None:
        # Convert the circle parameters a, b and r to integers.
        circles = np.uint16(np.around(circles))
        for pt in tqdm(circles[0, :]):
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(normalizedImgO, (a, b), r, (0, 0, 0), 2)
    print(circles)
    return normalizedImgO


def rescale(input):
    # rescale original 16 bit image to 8 bit values [0,255]
    x0 = input.min()
    x1 = input.max()
    y0 = 0
    y1 = 255.0
    i8 = ((input - x0) * ((y1 - y0) / (x1 - x0))) + y0
    # create new array with rescaled values and unsigned 8 bit data type
    o8 = i8.astype(np.uint8)
    return -o8


parser = argparse.ArgumentParser()
parser.add_argument('-input', dest='input', help='path to dicom directory', type=str)
results = parser.parse_args()

plt.imshow(Analyze(results.input))
plt.show()