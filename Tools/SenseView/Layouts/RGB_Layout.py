from PyQt5.QtWidgets import QHBoxLayout

from Widgets.RGBWidget import RGBWidget


class RGB_Layout(QHBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]

        self.rgb1 = RGBWidget(data_paths[0])
        self.addWidget(self.rgb1)

    def update_components(self,frame):
        self.rgb1.update(frame)

    def get_len(self):
        return self.rgb1.get_len()

    def get_time(self):
        return self.rgb1.get_time()