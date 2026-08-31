from PyQt5.QtWidgets import QHBoxLayout

from Widgets.GridEyeWidget import GridEyeWidget


class GridEye_Layout(QHBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]

        self.sensor = GridEyeWidget(data_paths[0])
        self.addWidget(self.sensor)

    def update_components(self,frame):
        self.sensor.update(frame)

    def get_len(self):
        return self.sensor.get_len()

    def get_time(self):
        return self.sensor.get_time()
