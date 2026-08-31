from pyqtgraph import ImageView

from Sensors.Camera import Camera


class RGBWidget(ImageView):
    def __init__(self, data_path):
        super().__init__()
        self.sensor = Camera()
        self.sensor_data = self.sensor.load_img_list(data_path)

    def update(self, frame):
        img = self.sensor.get_image(frame)
        self.setImage(img)
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.ui.histogram.hide()

    def get_len(self):
        return self.sensor_data.shape[0]

    def get_time(self):
        return self.sensor.get_time()