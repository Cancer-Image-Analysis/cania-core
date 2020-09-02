import cv2
import numpy as np
from enum import Enum

class StainingIntensityRange(object):
    def __init__(self, min_range: tuple, max_range: tuple):
        self._min = min_range
        self._max = max_range

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max


class ConfigurableStainingIntensityRange(StainingIntensityRange):
    def __init__(self, min_range: tuple, max_range: tuple):
        super().__init__(min_range, max_range)

    def set_min(self, min_range: tuple):
        self._min = min_range

    def set_max(self, max_range: tuple):
        self._max = max_range

class GenericStaining(object):
    def __init__(self, staining_name: str):
        self.name = staining_name

class ConfocalStaining(GenericStaining):
    def __init__(self, staining_name: str):
        super().__init__(staining_name)
        self.channel = -1

FM4_64 = ConfocalStaining('FM4-64')
PKH_67 = ConfocalStaining('PKH-67')
TUBULIN = ConfocalStaining('Tubulin')
CASPASE_3 = ConfocalStaining('Caspase-3')

'''
class CD8Filter(object):
    def __init__(self, hsv_min, hsv_max):
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max

    def filter_image(self, hsv_image, rgb_image):
        mask = cv2.inRange(hsv_image, self.hsv_min, self.hsv_max)
        mask_img = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=6)
        closed = cv2.bitwise_and(rgb_image, rgb_image, mask=closing)

        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        opened = cv2.bitwise_and(rgb_image, rgb_image, mask=opening)

        return mask_img, closed, opened

class MelanomaFilter(object):

    def __init__(self, hsv_min, hsv_max):
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max
        self.lysosome_filter = LysosomeFilter((1, 1, 1), (180, 255, 100))

    def filter_image(self, hsv_image, rgb_image):
        mask = cv2.inRange(hsv_image, self.hsv_min, self.hsv_max)
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        filtered_image = cv2.bitwise_and(rgb_image, rgb_image, mask=closing)
        
        #erosion = cv2.erode(closing, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 1)
        dilation = cv2.dilate(closing, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 10)
        melanoma_cell = cv2.bitwise_and(rgb_image, rgb_image, mask=dilation)
        cell_surface = np.bincount(dilation.ravel())[-1]

        lysosome_mask = self.lysosome_filter.filter_image(hsv_image, rgb_image)

        common_mask = cv2.bitwise_and(dilation, lysosome_mask)
        lysosome_surface = np.bincount(common_mask.ravel())[-1]

        print(lysosome_surface/(cell_surface- np.bincount(closing.ravel())[-1]))

        lysosomes = cv2.bitwise_and(rgb_image, rgb_image, mask=common_mask)

        #D = ndimage.distance_transform_edt(closing)
        #localMax = peak_local_max(D, indices=False, min_distance=30, labels=closing)
        #stainings = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        #labels = watershed(-D, stainings, mask=closing)
        #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        return filtered_image, melanoma_cell, lysosomes ##color.label2rgb(labels)

class InteractivChannelFilter(object):
    def __init__(self, rgb_images, hsv_images):
        fig, self.ax = plt.subplots(len(rgb_images), 4)
        plt.subplots_adjust(bottom=0.25)
        #self.ax.margins(x=0)

        self.original_rgb_images = rgb_images
        self.original_hsv_images = hsv_images
        self.filtered_images = rgb_images.copy()

        self.init_sliders()

        self.melanoma_nucleus = []
        self.melanoma_cells = []
        self.melanoma_lysosomes = []
        for i in range(len(rgb_images)):
            self.ax[i, 0].imshow(self.original_rgb_images[i])
            self.melanoma_nucleus.append(self.ax[i, 1].imshow(self.filtered_images[i]))
            self.melanoma_cells.append(self.ax[i, 2].imshow(self.filtered_images[i]))
            self.melanoma_lysosomes.append(self.ax[i, 3].imshow(self.filtered_images[i]))
        self.update(None)
        plt.show()

    def init_sliders(self):
        axcolor = 'lightgoldenrodyellow'
        ax_hue_min = plt.axes([0.12, 0.18, 0.78, 0.01], facecolor=axcolor)
        ax_hue_max = plt.axes([0.12, 0.15, 0.78, 0.01], facecolor=axcolor)

        ax_sat_min = plt.axes([0.12, 0.11, 0.78, 0.01], facecolor=axcolor)
        ax_sat_max = plt.axes([0.12, 0.08, 0.78, 0.01], facecolor=axcolor)

        ax_val_min = plt.axes([0.12, 0.04, 0.78, 0.01], facecolor=axcolor)
        ax_val_max = plt.axes([0.12, 0.01, 0.78, 0.01], facecolor=axcolor)

        # violet_filter = ColorFilterHSV((125, 75, 100), (145, 255, 255))

        self.slider_hue_min = Slider(ax_hue_min, 'Hue min', 0, 180, valinit=120, valstep=1)
        self.slider_hue_min.on_changed(self.update)
        self.slider_hue_max = Slider(ax_hue_max, 'Hue max', 0, 180, valinit=170, valstep=1)
        self.slider_hue_max.on_changed(self.update)

        self.slider_sat_min = Slider(ax_sat_min, 'Sat min', 0, 255, valinit=100, valstep=1)
        self.slider_sat_min.on_changed(self.update)
        self.slider_sat_max = Slider(ax_sat_max, 'Sat max', 0, 255, valinit=255, valstep=1)
        self.slider_sat_max.on_changed(self.update)

        self.slider_val_min = Slider(ax_val_min, 'Val min', 0, 255, valinit=100, valstep=1)
        self.slider_val_min.on_changed(self.update)
        self.slider_val_max = Slider(ax_val_max, 'Val max', 0, 255, valinit=255, valstep=1)
        self.slider_val_max.on_changed(self.update)

    def update(self, _):
        filter = CD8Filter((self.slider_hue_min.val, self.slider_sat_min.val, self.slider_val_min.val), 
        (self.slider_hue_max.val, self.slider_sat_max.val, self.slider_val_max.val))
        for i in range(len(self.filtered_images)):
            melanoma_nucleus, melanoma_cell, melanoma_lysosome = filter.filter_image(self.original_hsv_images[i], self.original_rgb_images[i])
            self.melanoma_nucleus[i].set_data(melanoma_nucleus)
            self.melanoma_cells[i].set_data(melanoma_cell)
            self.melanoma_lysosomes[i].set_data(melanoma_lysosome)


if __name__ == "__main__" :
    rgb_1, hsv_1 = open_slide_rgb_and_hsv('/home/twarz/Projects/immuno-deconv/src/data/mrxs/NVA_MELANLisaV7_15T005491.03.P4595.mrxs', 43000, 99000)
    rgb_2, hsv_2 = open_slide_rgb_and_hsv('/home/twarz/Projects/immuno-deconv/src/data/mrxs/NVA_MELANLisaV7_15T006895.02.P4595.mrxs', 43000, 122000)
    rgb_3, hsv_3 = open_slide_rgb_and_hsv('/home/twarz/Projects/immuno-deconv/src/data/mrxs/NVA_MELANLisaV7_15T032948.01.P4595.mrxs', 43000, 122000)
    rgb_4, hsv_4 = open_slide_rgb_and_hsv('/home/twarz/Projects/immuno-deconv/src/data/mrxs/NVA_MELANLisaV7_16T024341.01.P4595.mrxs', 43000, 122000)
    rgb_5, hsv_5 = open_slide_rgb_and_hsv('/home/twarz/Projects/immuno-deconv/src/data/mrxs/NVA_MELANLisaV7_16T024341.01.P4595.mrxs', 36000, 122000)
    
    rgb_images = np.array([rgb_1, rgb_2])#, rgb_3, rgb_4, rgb_5])
    hsv_images = np.array([hsv_1, hsv_2])#, hsv_3, hsv_4, hsv_5])

    filter = InteractivChannelFilter(rgb_images, hsv_images)
'''