from pyqtgraph import ImageView

from Sensors.Camera import Camera
from Sensors.Wildcam2 import Wildcam2


class Wildcam2Widget(ImageView):
    def __init__(self, image_folder,csv_path):
        super().__init__()
        self.sensor = Wildcam2(image_folder,csv_path)
        self.sensor_data = self.sensor.imgs_pd

    def update(self, frame):
        img = self.sensor.get_overlay(frame)
        self.setImage(img)
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.ui.histogram.hide()

    def get_len(self):
        return self.sensor_data.shape[0]

    def get_time(self):
        return self.sensor.get_time()