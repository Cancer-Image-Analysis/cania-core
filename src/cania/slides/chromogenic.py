import numpy as np
import cv2
from enum import Enum

from cania.slides.stainings import StainingIntensityRange, ConfigurableStainingIntensityRange, GenericStaining
from cania.slides.slides import GenericSlide
from cania.slides.regions import ScannerRegionData, SlideRegion


class ChromogenicSlide(GenericSlide):
    def __init__(self, slide_path: str, slide_id: str):
        super().__init__(slide_path, slide_id)

    def get_region(self, x: int, y: int, width: int, height: int, level: int):
        pil_region = self.slide.read_region((x, y), level, (width, height))
        region_image = np.array(pil_region)[:, :, :3]  # remove alpha
        # region_image is RGB but BGR for cv2...
        region_data = ScannerRegionData(x, y, width, height, level)
        return SlideRegion(region_data, region_image)

    def get_stainings(self, slide_region) -> dict:
        rgb = cv2.cvtColor(slide_region.get_image(), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        stainings = dict()
        stainings['rgb'] = rgb
        stainings['hsv'] = hsv
        for staining_name in self.stainings.keys():
            stainings[staining_name] = self.stainings[staining_name].get_mask(hsv)
        return stainings

    def get_masked_region(self, mask, region_image):
        return cv2.bitwise_and(region_image, region_image, mask=mask)


class ColorFilterHSV(object):
    def __init__(self, hsv_range: StainingIntensityRange):
        self.hsv_range = hsv_range

    def get_mask(self, hsv_image) -> np.ndarray:
        mask = cv2.inRange(hsv_image, self.hsv_range.get_min(), self.hsv_range.get_max())
        return mask


class ConfigurableColorFilterHSV(ColorFilterHSV):
    def __init__(self):
        super(ConfigurableColorFilterHSV, self).__init__(ConfigurableStainingIntensityRange((0, 0, 0), (180, 255, 255)))


class ChromogenicStaining(GenericStaining):
    def __init__(self, color_filter: ColorFilterHSV, staining_name: str):
        super().__init__(staining_name)
        self.color_filter = color_filter

    def get_mask(self, hsv_image) -> np.ndarray:
        return self.color_filter.get_mask(hsv_image)


class ConfigurableChromogenicStaining(ChromogenicStaining):
    def __init__(self, name='configurable'):
        super().__init__(ConfigurableColorFilterHSV(), name)

    def configure(self, min_range, max_range):
        self.color_filter.hsv_range.set_min(min_range)
        self.color_filter.hsv_range.set_max(max_range)


class ConfigurableChromogenicSlide(ChromogenicSlide):
    def __init__(self, slide_path: str, slide_id: str):
        super().__init__(slide_path, slide_id)
        self.add_staining(ConfigurableChromogenicStaining())

    def configure(self, min_range, max_range):
        self.stainings['configurable'].configure(min_range, max_range)


class StainingColor(Enum):
    BLACK  = StainingIntensityRange((0,  0,   0),   (180, 100, 100))
    ORANGE = StainingIntensityRange((15,  130, 120), (30,  255, 255))
    PURPLE = StainingIntensityRange((110, 60, 70), (160, 255, 255))
    BLUE   = StainingIntensityRange((85, 60,  120), (110, 255, 255))
    BROWN  = StainingIntensityRange((0,   100, 0),   (15,  255, 255))


LAMP1    = ChromogenicStaining(ColorFilterHSV(StainingColor.BLACK.value), 'LAMP1')
CD107A   = ChromogenicStaining(ColorFilterHSV(StainingColor.BLACK.value), 'CD107a')
SOX10    = ChromogenicStaining(ColorFilterHSV(StainingColor.ORANGE.value), 'Sox10')
CD8      = ChromogenicStaining(ColorFilterHSV(StainingColor.PURPLE.value), 'CD8')
BLUE     = ChromogenicStaining(ColorFilterHSV(StainingColor.BLUE.value), 'BLUE')
MELANINE = ChromogenicStaining(ColorFilterHSV(StainingColor.BROWN.value), 'Melanine')
