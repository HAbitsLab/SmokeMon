from pyqtgraph import ImageView

from Sensors.MLX import MLX


class MLXWidget(ImageView):
    def __init__(self, data_path):
        super().__init__()
        self.sensor = MLX()
        self.sensor_data = self.sensor.load_csv(data_path)
        self.once = False

    def get_len(self):
        return self.sensor_data.shape[0]


    def update(self, frame):
        img = self.sensor.get_image(frame)
        img2 = self.sensor.get_image(frame-1)
        self.setImage(img-img2)
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.ui.histogram.hide()

    def update_human(self, frame):
        img = self.sensor.get_human(frame)
        self.setImage(img)
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.ui.histogram.hide()

    def update_history(self, frame):
        img = self.sensor.get_history(frame)
        self.setImage(img)
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.ui.histogram.hide()

    def get_time(self):
        return self.sensor.get_time()