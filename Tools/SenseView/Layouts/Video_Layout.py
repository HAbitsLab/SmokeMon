from PyQt5.QtWidgets import QHBoxLayout

from Widgets.VideoWidget import VideoWidget


class Video_Layout(QHBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]

        self.video = VideoWidget(data_paths[0],4)
        self.addWidget(self.video)

    def update_components(self,frame):
        self.video.update(frame)

    def get_len(self):
        return self.video.get_len()

    def get_time(self):
        return self.video.get_time()