
from pathlib import Path

from openslide import open_slide

from cania.slides.exceptions import EmptySlideIdError

class SlideFactory(object):
    @staticmethod
    def open_slide(slide_path: str):
        slide_file = Path(slide_path)
        print(slide_file)
        if not slide_file.is_file():
            raise FileNotFoundError

        extension = slide_file.suffix
        if extension in ['.mrxs']:
            slide = open_slide(slide_path)
        elif extension in ['.tif', '.lsm']:
            slide = tifffile.imread(slide_path)
            logger.info('array shape: ' + str(slide.shape)) 
            slide = slide / np.repeat(np.max(np.max(slide, axis=1), axis=1), 77*77).reshape(5, 77, 77) * 255
            slide = slide.astype(np.uint8)
            #cv2.normalize(slide,  slide, 0, 255, cv2.NORM_MINMAX)
        elif extension in ['.czi']:
            slide = czifile.imread(slide_path)
        else:
            raise FileNotFoundError
        return slide

class GenericSlide(object):
    def __init__(self, slide_path: str, slide_id: str):
        if not slide_id:
            raise EmptySlideIdError
        self.slide_id = slide_id
        self.slide = SlideFactory.open_slide(slide_path)
        self.stainings = {}

    def add_staining(self, staining):
        self.stainings[staining.name] = staining

    def clean(self):
        self.stainings = {}