from pyqtgraph import ImageView

from Sensors.Grideye import Grideye


class GridEyeWidget(ImageView):
    def __init__(self, data_path):
        super().__init__()
        self.sensor = Grideye(data_path)
        self.sensor_data = self.sensor.load_csv(data_path)
        self.once = False

    def get_len(self):
        return self.sensor_data.shape[0]

    def get_diff_pd(self):
        return self.sensor.get_diff_pd()


    def update(self, frame):

        img = self.sensor.get_image(frame)
        self.setImage(img)
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.ui.histogram.hide()

    def get_time(self):
        return self.sensor.get_time()