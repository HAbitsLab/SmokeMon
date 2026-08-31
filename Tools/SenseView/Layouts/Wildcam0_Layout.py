from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
import pandas as pd
import numpy as np

from Widgets.GridEyeWidget import GridEyeWidget
from Widgets.RGBWidget import RGBWidget
from Widgets.TimeSeriesWidget import TimeSeriesWidget
from Widgets.Wildcam0Widget import Wildcam0Widget


class Wildcam0_Layout(QVBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]

        h_layout = QHBoxLayout()

        self.rgb = RGBWidget(data_paths[0])
        h_layout.addWidget(self.rgb)

        self.sensor = GridEyeWidget(data_paths[1])
        # h_layout.addWidget(self.sensor)

        self.obf = Wildcam0Widget(data_paths[0],data_paths[1])
        h_layout.addWidget(self.obf)

        # ts_temp = pd.DataFrame(data=np.ones((self.sensor.get_len(), 2)))
        # ts_temp = self.sensor.get_diff_pd()
        # ts_temp.to_csv(data_paths[2])
        #
        # self.ts = TimeSeriesWidget(ts_temp)
        # self.addWidget(self.ts)

        self.addLayout(h_layout)

    def update_components(self, frame):
            # self.sensor.update(frame)
            self.rgb.update(frame)
            self.obf.update(frame)
            # self.ts.update(frame)

    def get_len(self):
        return self.sensor.get_len()

    def get_time(self):
        return self.sensor.get_time()
