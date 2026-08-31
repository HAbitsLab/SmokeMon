from PyQt5.QtWidgets import QHBoxLayout

from Widgets.MLXWidget import MLXWidget


class MLX_Layout(QHBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]

        self.mlx = MLXWidget(data_paths[0])
        self.addWidget(self.mlx)

        # self.human = MLXWidget(data_paths[0])
        # self.addWidget(self.human)

        # self.blur= MLXWidget(data_paths[0])
        # self.addWidget(self.blur)

    def update_components(self,frame):
        self.mlx.update(frame)
        # self.human.update_human(frame)
        # self.blur.update_blur(frame)

    def get_len(self):
        return self.mlx.get_len()

    def get_time(self):
        return self.mlx.get_time()
