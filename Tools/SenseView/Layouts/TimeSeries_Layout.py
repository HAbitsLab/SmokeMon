from PyQt5.QtWidgets import QHBoxLayout

from Widgets.TimeSeriesWidget import TimeSeriesWidget


class TimeSeries_Layout(QHBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]

        self.ts = TimeSeriesWidget(data_paths[0])
        self.addWidget(self.ts)

    def update_components(self,frame):
        self.ts.update(frame)

    def get_len(self):
        return self.ts.get_len()

    def get_time(self):
        return self.ts.get_time()
