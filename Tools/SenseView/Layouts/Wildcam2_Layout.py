from PyQt5.QtWidgets import QHBoxLayout

from Widgets.MLXWidget import MLXWidget
from Widgets.RGBWidget import RGBWidget
from Widgets.Wildcam2Widget import Wildcam2Widget


class Wildcam2_Layout(QHBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]


        self.rgb = RGBWidget(data_paths[0])
        self.addWidget(self.rgb)

        self.mlx = MLXWidget(data_paths[1])
        self.addWidget(self.mlx)

        self.wildcam = Wildcam2Widget(data_paths[0],data_paths[1])
        self.addWidget(self.wildcam)



    def update_components(self,frame):
        self.mlx.update(frame)
        self.rgb.update(frame)
        self.wildcam.update(frame)

    def get_len(self):
        return self.mlx.get_len()

    def get_time(self):
        return self.mlx.get_time()
