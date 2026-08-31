from pyqtgraph import ImageView

from Sensors.Video import Video


class VideoWidget(ImageView):
    def __init__(self, data_path,fps):
        super().__init__()
        self.sensor = Video()
        self.sensor_data = self.sensor.load_data(data_path,fps)

    def get_len(self):
        return self.sensor.total_frames

    def update(self, frame):
        img = self.sensor.get_image(frame)
        self.setImage(img)
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.ui.histogram.hide()

    def get_time(self):
        return self.sensor.get_time()