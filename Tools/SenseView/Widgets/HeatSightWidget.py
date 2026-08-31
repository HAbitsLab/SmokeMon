from pyqtgraph import ImageView

from Sensors.HeatSight import HeatSight


class HeatSightWidget(ImageView):
    def __init__(self, data_path,transform=False):
        super().__init__()
        self.sensor = HeatSight()
        self.sensor_data = self.sensor.load_csv(data_path)
        self.sensor.transform = transform

    def get_len(self):
        return self.sensor_data.shape[0]


    def update(self, frame):
        img = self.sensor.get_image(frame)
        self.setImage(img,autoLevels=False)
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.ui.histogram.hide()

    def get_time(self):
        return self.sensor.get_time()