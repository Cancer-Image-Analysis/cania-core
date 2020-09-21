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


def write_bgr(filename, rgb_image):
    cv2.imwrite(filename, rgb2bgr(rgb_image))


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


def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


""" channels """


def split_channels(image):
    return list(cv2.split(image))


""" draw on images """


def overlay(img, mask, color=[255, 255, 0], alpha=0.4):
    # Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    out = img.copy()
    img_layer = img.copy()
    print(mask.shape)
    img_layer[np.where(mask)] = color
    overlayed = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlayed, contours, -1, color, 2)
    return overlayed


def fill_ellipses(mask, ellipses):
    for ellipse in ellipses:
        cv2.ellipse(mask, ellipse, 1, thickness=-1)
    return mask


""" operations """

def resize(img, scale):
    return cv2.resize(img, scale)


def imfill(img):
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    im_floodfill = img.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = img | im_floodfill_inv

    return im_out