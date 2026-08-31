from pyqtgraph import PlotWidget, InfiniteLine
import numpy as np

from Sensors.TimeSeries import TimeSeries


class TimeSeriesWidget(PlotWidget):
    def __init__(self, data_path):
        super().__init__()

        self.sensor = TimeSeries(data_path)
        self.sensor_data = self.sensor.data



        self.p1 = self.plot(self.sensor_data.values[:,1])
        self.p1.setPen((0, 0, 255),width=4)
        self.setYRange(-1, 1)
        self.setMaximumHeight(100)
        self.setMouseEnabled(y=False)
        self.ax_b = self.getAxis('bottom')  # This is the trick
        self.ax_b .setTicks([])

        self.time_cursor = InfiniteLine(angle=90)
        self.addItem(self.time_cursor)

        self.viewbox_size = 100

        # # self.scene().sigMouseClicked.connect(self.mouse_clicked)
        # self.scene()..connect(self.mouse_clicked)


    def get_len(self):
        return self.sensor_data.shape[0]


    def update(self, index):
        self.time_cursor.setValue(index)
        self.setXRange(index-self.viewbox_size , index+self.viewbox_size)
        # win = self.sensor.get_window(index)
        # self.p1.setData(y=win.values[:,1], x=win.index.values)
        # print("updating")

    def get_time(self):
        return self.sensor.get_time()

    def wheelEvent(self, event):
        super(TimeSeriesWidget, self).wheelEvent(event)
        self.viewbox_size = np.ceil((self.ax_b.range[1]-self.ax_b.range[0])/2)