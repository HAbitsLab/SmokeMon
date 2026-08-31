from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
import pandas as pd
import numpy as np

from Widgets.GridEyeWidget import GridEyeWidget
from Widgets.RGBWidget import RGBWidget
from Widgets.TimeSeriesWidget import TimeSeriesWidget
from Widgets.Wildcam1Widget import Wildcam1Widget


class Wildcam1_Layout(QVBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]

        h_layout = QHBoxLayout()

        self.rgb = RGBWidget(data_paths[0])
        h_layout.addWidget(self.rgb)

        self.sensor = GridEyeWidget(data_paths[1])
        h_layout.addWidget(self.sensor)

        self.obf = Wildcam1Widget(data_paths[0],data_paths[1])
        h_layout.addWidget(self.obf)


        self.addLayout(h_layout)

    def update_components(self, frame):
            self.sensor.update(frame)
            self.rgb.update(frame)
            self.obf.update(frame)


    def get_len(self):
        return self.sensor.get_len()

    def get_time(self):
        return self.sensor.get_time()
