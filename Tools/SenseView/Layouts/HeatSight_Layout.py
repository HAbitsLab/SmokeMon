from PyQt5.QtWidgets import QHBoxLayout, QGridLayout


from Widgets.HeatSightWidget import HeatSightWidget


class HeatSight_Layout(QHBoxLayout):

    def __init__(self, settings):
        super().__init__()

        data_paths = settings["data_paths"]

        self.sensor = HeatSightWidget(data_paths[0])
        self.addWidget(self.sensor)


        # self.sensor_trigger = HeatSightWidget(data_paths[1])
        # self.addWidget(self.sensor_trigger)

        # self.sensor_normalized = HeatSightWidget(data_paths[0], transform=True)
        # self.addWidget(self.sensor_normalized)






    def update_components(self,frame):
        self.sensor.update(frame)
        # self.sensor_trigger.update(frame)


    def get_len(self):
        return self.sensor.get_len()

    def get_time(self):
        return self.sensor.get_time()