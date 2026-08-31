from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout

from Widgets.MLXWidget import MLXWidget
from Widgets.TimeSeriesWidget import TimeSeriesWidget
from Widgets.VideoWidget import VideoWidget


class SmokeMon_Layout(QVBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]
        self.sync = settings["sync"]
        self.with_video = True
        self.with_ts = False

        h_layout = QHBoxLayout()

        if data_paths[0] == "":
            self.with_video = False

        if data_paths[2] == "":
            self.with_ts = False

        if self.with_video:
            self.video = VideoWidget(data_paths[0], 4)
            h_layout.addWidget(self.video)

        self.mlx = MLXWidget(data_paths[1])
        h_layout.addWidget(self.mlx)

        if self.with_ts:
            self.ts = TimeSeriesWidget(data_paths[2])
            self.addWidget(self.ts)

        self.addLayout(h_layout)

    def update_components(self, frame):
        self.mlx.update(frame + self.sync["mlx"])
        if self.with_video:
            self.video.update(frame)
        if self.with_ts:
            self.ts.update(frame + self.sync["cp"])

    def get_len(self):
        return self.mlx.get_len()

    def get_time(self):
        return self.mlx.get_time()

