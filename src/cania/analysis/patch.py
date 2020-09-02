from abc import ABC, abstractmethod
import numpy as np
import cv2


class PatchGenerator(ABC):
    def __init__(self, patch_size):
        self._patch_size = patch_size

    @abstractmethod
    def get_patches(self, slide, all_indices):
        pass


class PatchPaddingIHC(object):
    def __init__(self, stainings, real_x, real_y, size):
        self.stainings = stainings
        self.real_x = real_x
        self.real_y = real_y
        self.size = size


class PatchGeneratorIHC(PatchGenerator):
    def __init__(self, patch_size, patch_level):
        super(PatchGeneratorIHC, self).__init__(patch_size)
        self._patch_level = patch_level

    def get_patches(self, ihc_slide, all_indices):
        # pad if possible
        for i in range(len(all_indices)):
            x = all_indices[i][1]
            y = all_indices[i][0]
            yield ihc_slide.get_region(x, y, self._patch_size, self._patch_size, self._patch_level)


class PatchGeneratorWithPaddingIHC(PatchGeneratorIHC):
    '''
    add padding around patch to avoid the analysis being distorted by the edge
    of the image usefull when doing image processing
    '''

    def __init__(self, size, level, padding):
        super(PatchGeneratorWithPaddingIHC, self).__init__(size, level)
        self._padding = padding

    def get_patches(self, ihc_slide, all_indices):
        min_all_indices = all_indices-self._padding
        max_all_indices = all_indices+self._padding+self._patch_size
        w_max, h_max = ihc_slide.slide.level_dimensions[self._patch_level]
        # pad if possible
        for i in range(len(min_all_indices)):
            x_min = max(0, min_all_indices[i][1])
            y_min = max(0, min_all_indices[i][0])
            x_max = min(w_max, max_all_indices[i][1])
            y_max = min(h_max, max_all_indices[i][0])
            w = x_max-x_min
            h = y_max-y_min
            real_x = all_indices[i][1] - x_min
            real_y = all_indices[i][0] - y_min
            patch_region = ihc_slide.get_region(x_min, y_min, w, h, self._patch_level)
            patch_stainings = ihc_slide.get_stainings(patch_region)

            yield PatchPaddingIHC(patch_stainings, real_x, real_y, self._patch_size)


class GridPatchAnalysis(ABC):
    def __init__(self, patch_generator, patch_analysis):
        self.patch_generator = patch_generator
        self.patch_analysis = patch_analysis

    def run(self, ihc_slide, all_indices):
        results = []  # row list
        melanoma = np.zeros((779, 351)).astype(np.uint8)
        CD8 = np.zeros((779, 351)).astype(np.uint8)
        for patch, indices in zip(self.patch_generator.get_patches(ihc_slide, all_indices), all_indices/256):
            x, y = int(indices[1]), int(indices[0])
            result = self.patch_analysis.run(patch)
            melanoma[y, x] = int(result['melanoma.n'])
            CD8[y, x] = int(result['CD8.n'])
            results.append(result)
        cv2.imwrite('melanoma_map.png', cv2.applyColorMap(melanoma, cv2.COLORMAP_VIRIDIS))
        cv2.imwrite('CD8_map.png', cv2.applyColorMap(CD8, cv2.COLORMAP_VIRIDIS))
        return results
