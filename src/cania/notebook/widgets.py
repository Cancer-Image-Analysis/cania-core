
import ipywidgets as w
from IPython.display import display, Image
from matplotlib import pyplot as plt
from io import BytesIO
import PIL
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
import time

'''
class ImageViewer(widgets.Image):
    def __init__(self, np_image):
        self.update(np_image)

    def update(self, np_image):
        im = PIL.Image.fromarray(np_image)
        self.bytes_images = BytesIO()
        im.save(self.bytes_images, format='png')

    def display(self):
        display(Image(self.bytes_images.getvalue(), format='png', retina=True))
'''

class SlideProcessInfo(object):
    def __init__(self, slide_path, name):
        layout=w.Layout(padding='0 50px 0 0')
        self.progress = w.FloatProgress(
            value=0.,
            min=0.,
            max=1.,
            step=0.001,
            description='Process:',
            bar_style='info',
            orientation='horizontal',
            layout=layout
        )
        self.slide_path = slide_path
        self.name = w.Label(value=name, layout=w.Layout(width='150px', padding='0 50px 0 0'))
        self.state = w.Label(value='WAIT', layout=w.Layout(width='150px', padding='0 50px 0 0'))
        self.time = w.Label(value='--:--', layout=layout)
        self.widget_line = w.HBox([self.name, self.state, self.progress, self.time], layout=layout)
        self.init_time = None
        
    def start(self):
        self.init_time = time.time()
        self.state.value = 'WORKING'
       
    def update_time(self):
        self.time.value = time.strftime('%M:%S', time.gmtime(time.time()-self.init_time))
        
    def update_progress(self, progress):
        self.progress.value = progress
        
    def end(self, state='DONE', flag='success'):
        self.state.value = state
        self.progress.bar_style=flag
        self.update_progress(1.)
        self.update_time()

class SlideRegionViewer(object):
    '''
    https://stackoverflow.com/questions/52238567/have-2-ipywidgets-acting-on-one-matplotlib-plot-in-jupyter-python
    '''
    def __init__(self, slide, default_x, default_y, default_w, default_h, default_l=0, **kwargs):
        self.slide = slide
        self.slide_region = self.slide.get_region(default_x, default_y, default_w, default_h, default_l)
        self.output_widget = w.Output()  # will contain the plot
        self.container = w.VBox(**kwargs)  # Contains the whole app
        self.container.layout.align_items = 'center'
        self.draw_app(default_x, default_y, default_w, default_h, default_l)

    def draw_app(self, default_x, default_y, default_w, default_h, default_l=0):
        """
        Draw the sliders and the output widget

        This just runs once at app startup.
        """
        self.sliders = dict()
        self.sliders['x'] = w.IntSlider(value=default_x, min=0, max=90000, step=64, description='x: ', continuous_update=False)
        self.sliders['y'] = w.IntSlider(value=default_y, min=0, max=200000, step=64, description='y: ', continuous_update=False)
        self.sliders['w'] = w.IntSlider(value=default_w, min=64, max=8192, step=64, description='width: ', continuous_update=False)
        self.sliders['h'] = w.IntSlider(value=default_h, min=64, max=8192, step=64, description='height: ', continuous_update=False)
        self.sliders['l'] = w.IntSlider(value=default_l, min=0, max=8, step=1, description='level: ', continuous_update=False)

        self.slider_line = w.HBox([w.VBox([self.sliders['x'], self.sliders['y']]), w.VBox([self.sliders['w'], self.sliders['h']]), self.sliders['l']])
        self.out = w.interactive_output(self.update_region, self.sliders)

        self.container.children = [self.slider_line, self.output_widget]

    def update_region(self, x, y, w, h, l):
        self.slide_region = self.slide.get_region(x, y, w, h, l)
        print('update')
        self.output_widget.clear_output(wait=True)
        with self.output_widget as f:
            plt.figure(figsize=(12, 12), dpi=80)
            plt.axis('off')
            plt.imshow(self.slide_region.get_image())
            plt.show()


class MarkerConfiguration(object):
    def __init__(self, slide, region_updater, h=[0, 180], s=[0, 255], v=[0, 255], **kwargs):
        self.slide = slide
        self.region_updater = region_updater
        self.output_widget = w.Output()  # will contain the plot
        self.container = w.VBox(**kwargs)  # Contains the whole app
        self.container.layout.align_items = 'center'
        self.draw_app(h, s, v)

    def draw_app(self, h, s, v):
        """
        Draw the sliders and the output widget

        This just runs once at app startup.
        """
        self.sliders = dict()
        self.sliders['h'] = w.IntRangeSlider(value=h, min=0, max=180, step=1, description='Hue: ',continuous_update=False)
        self.sliders['s'] = w.IntRangeSlider(value=s, min=0, max=255, step=1, description='Sat: ',continuous_update=False)
        self.sliders['v'] = w.IntRangeSlider(value=v, min=0, max=255, step=1, description='Val: ',continuous_update=False)

        self.slider_line = w.HBox([self.sliders['h'], self.sliders['s'], self.sliders['v']])
        self.out = w.interactive_output(self.update_marker, self.sliders)

        self.container.children = [self.slider_line, self.output_widget]
        self.update_marker(self.sliders['h'].value, self.sliders['s'].value, self.sliders['v'].value)

    def update_marker(self, h, s, v):
        print('update')
        h_min, h_max = h
        s_min, s_max = s
        v_min, v_max = v
        self.slide.configure((h_min, s_min, v_min), (h_max, s_max, v_max))
        stainings = self.slide.get_stainings(self.region_updater.slide_region)
        mask = stainings['configurable']
        self.new_image = self.slide.get_masked_region(mask, self.region_updater.slide_region.get_image())
        self.output_widget.clear_output(wait=True)
        with self.output_widget as f:
            plt.figure(figsize=(12, 12), dpi=80)
            plt.axis('off')
            plt.imshow(self.new_image)
            plt.show()


class StainingViewer(object):
    def __init__(self, slide, region_updater, **kwargs):
        self.slide = slide
        self.region_updater = region_updater
        self.output_widget = w.Output()  # will contain the plot
        self.container = w.VBox([self.output_widget], **kwargs)  # Contains the whole app
        self.container.layout.align_items = 'center'

    def update_region(self, x, y, w, h, l):
        self.slide_region = self.slide.get_region(x, y, w, h, l)
        masks, hsv_img = self.slide.get_masks_on_region(self.slide_region)
        self.output_widget.clear_output(wait=True)
        h, s, v = np.indices((180, 255, 255))
        with self.output_widget as f:
            fig, axs = plt.subplots(3, 2, figsize=(14, 14), dpi=80)
            plt.axis('off')
            axs[0, 0].imshow(self.slide.get_masked_region(masks['Melanine'], self.region_updater.slide_region.get_image()))
            axs[0, 1].imshow(self.slide.get_masked_region(masks['LAMP1'], self.region_updater.slide_region.get_image()))
            axs[1, 0].imshow(self.slide.get_masked_region(masks['Sox10'], self.region_updater.slide_region.get_image()))
            axs[1, 1].imshow(self.slide.get_masked_region(masks['CD8'], self.region_updater.slide_region.get_image()))
            axs[2, 0].imshow(self.slide.get_masked_region(masks['BLUE'], self.region_updater.slide_region.get_image()))
            '''
            orange = (h < 30) & (s < 255) & (v < 255) & (h > 18) & (s > 120) & (v > 100)
            black = (h < 180) & (s < 255) & (v < 100) & (h > 18) & (s > 0) & (v > 0)
            brown = (h < 18) & (s < 255) & (v < 255) & (h > 0) & (s > 0) & (v > 100)
            blue = (h < 114) & (s < 255) & (v < 255) & (h > 100) & (s > 60) & (v > 100)
            purple = (h < 180) & (s < 255) & (v < 255) & (h > 114) & (s > 160) & (v > 0)

            voxels = orange | black | brown | blue | purple
            colors = np.empty(voxels.shape, dtype=object)
            colors[orange] = 'orange'
            colors[black] = 'black'
            colors[brown] = 'chocolate'
            colors[blue] = 'azure'
            colors[purple] = 'purple'
            ax = fig.gca(projection='3d')
            ax.voxels(voxels, facecolors=colors)
            '''
            plt.show()



class ConfigureLAMP1(object):
    def __init__(self, original_image, t1, t2, t3, **kwargs):
        self.original_image = (original_image*255).astype(np.uint8)
        _, self.tumor_mask = cv2.threshold(self.original_image, 0., 255, cv2.THRESH_BINARY)
        #self.tumor_mask = self.tumor_mask.astype(np.uint8)
        self.output_widget = w.Output()  # will contain the plot
        self.container = w.VBox(**kwargs)  # Contains the whole app
        self.container.layout.align_items = 'center'
        self.draw_app(t1, t2, t3)
        self.update_thresholds(t1, t2, t3)
        print('created')
        
    def draw_app(self, t1, t2, t3):
        """
        Draw the sliders and the output widget

        This just runs once at app startup.
        """
        self.sliders = dict()
        self.sliders['t1'] = w.FloatSlider(value=t1, min=0., max=1., step=0.01, description='T1: ', continuous_update=False)
        self.sliders['t2'] = w.FloatSlider(value=t2, min=0., max=1., step=0.01, description='T2: ', continuous_update=False)
        self.sliders['t3'] = w.FloatSlider(value=t3, min=0., max=1., step=0.01, description='T3: ', continuous_update=False)

        self.slider_line = w.HBox([self.sliders['t1'], self.sliders['t2'], self.sliders['t3']])
        self.out = w.interactive_output(self.update_thresholds, self.sliders)

        self.container.children = [self.slider_line, self.output_widget]

    def update_thresholds(self, t1, t2, t3):
        _, new_mask_t1 = cv2.threshold(self.original_image, int(round(t1*255)), 255, cv2.THRESH_BINARY_INV)
        t1_only_tumor = cv2.bitwise_and(new_mask_t1, new_mask_t1, mask=self.tumor_mask)
        surface_t1 = np.bincount(t1_only_tumor.ravel())[-1]

        _, new_mask_t2 = cv2.threshold(self.original_image, int(round(t2*255)), 255, cv2.THRESH_BINARY_INV)
        t2_without_t1 = cv2.bitwise_and(new_mask_t2, cv2.bitwise_not(new_mask_t1), mask=self.tumor_mask)
        surface_t2 = np.bincount(t2_without_t1.ravel())[-1]

        _, new_mask_t3 = cv2.threshold(self.original_image, int(round(t3*255)), 255, cv2.THRESH_BINARY_INV)
        t3_without_t2 = cv2.bitwise_and(new_mask_t3, cv2.bitwise_not(new_mask_t2), mask=self.tumor_mask)
        surface_t3 = np.bincount(t3_without_t2.ravel())[-1]

        _, new_mask_max = cv2.threshold(self.original_image, 254, 255, cv2.THRESH_BINARY_INV)
        max_without_t2 = cv2.bitwise_and(new_mask_max, cv2.bitwise_not(new_mask_t3), mask=self.tumor_mask)
        surface_max = np.bincount(max_without_t2.ravel())[-1]

        print(surface_t1, surface_t2, surface_t3, surface_max)
        print(surface_t1 + surface_t2 + surface_t3 + surface_max)

        print(np.bincount(self.tumor_mask.ravel())[-1])

        self.output_widget.clear_output(wait=True)
        with self.output_widget as f:
            fig, axs = plt.subplots(2, 4, figsize=(14, 14), dpi=80)
            axs[0, 0].imshow(new_mask_t1)
            axs[0, 0].set_axis_off()
            axs[0, 0].set_title('0% to ' + str(t1*100) + '%')
            axs[0, 1].imshow(new_mask_t2)
            axs[0, 1].set_axis_off()
            axs[0, 1].set_title(str(t1*100) + '% to ' + str(t2*100) + '%')
            axs[0, 2].imshow(new_mask_t3)
            axs[0, 2].set_axis_off()
            axs[0, 2].set_title(str(t2*100) + '% to ' + str(t3*100) + '%')
            axs[0, 3].imshow(new_mask_max)
            axs[0, 3].set_axis_off()
            axs[0, 3].set_title(str(t3*100) + '% to 100%')

            axs[1, 0].imshow(t1_only_tumor)
            axs[1, 0].set_axis_off()
            axs[1, 0].set_title('0% to ' + str(t1*100) + '%')
            axs[1, 1].imshow(t2_without_t1)
            axs[1, 1].set_axis_off()
            axs[1, 1].set_title(str(t1*100) + '% to ' + str(t2*100) + '%')
            axs[1, 2].imshow(t3_without_t2)
            axs[1, 2].set_axis_off()
            axs[1, 2].set_title(str(t2*100) + '% to ' + str(t3*100) + '%')
            axs[1, 3].imshow(max_without_t2)
            axs[1, 3].set_axis_off()
            axs[1, 3].set_title(str(t3*100) + '% to 100%')
            plt.show()