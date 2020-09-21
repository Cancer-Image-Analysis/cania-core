import tifffile
import cv2
import numpy as np

""" read images """


def read_rgb(filename):
    return cv2.imread(filename, cv2.IMREAD_COLOR)


def read_gray(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def read_mirax(filename):
    pass


def read_lsm(filename):
    pass


def read_tiff(filename):
    return tifffile.imread(filename)


""" write images """


def write_rgb(filename, rgb_image):
    cv2.imwrite(filename, rgb_image)


def write_tiff():
    pass


def write_gray():
    pass


""" new image """


def new_image(shape):
    return np.zeros(shape=shape, dtype=np.uint8)


""" color conversion """


def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


""" channels """


def split_channels(image):
    return list(cv2.split(image))


""" draw on images """


def overlay(img, mask, color=[255, 255, 0], alpha=0.4):
    # Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    out = img.copy()
    img_layer = img.copy()
    img_layer[np.where(mask)] = color
    overlayed = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlayed, contours, -1, color, 2)
    return overlayed


def fill_ellipses(mask, ellipses):
    for ellipse in ellipses:
        cv2.ellipse(mask, ellipse, 1, thickness=-1)
    return mask
