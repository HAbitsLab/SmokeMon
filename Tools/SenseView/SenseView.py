import os
import numpy as np
import pandas as pd
from datetime import datetime

import yaml
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QMainWindow, QApplication, QSlider, QLabel

import pyqtgraph as pg

import argparse

from Layouts import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', default="/Users/rma145/Desktop/thermo-smoking/Data/Wild_structured/P0/clean/senseView.yaml",
                    type=str, required=True, help='path to input (image folder or CSV)')
parser.add_argument('-l', default="SmokeMon_Layout", type=str, required=True, help='layout name')


class Color(QWidget):

    def __init__(self, color, *args, **kwargs):
        super(Color, self).__init__(*args, **kwargs)
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


class SenseView(QMainWindow):
    frame = 0
    max_frames = 20000

    def __init__(self, args):
        super().__init__()

        self.window = QWidget()
        self.setCentralWidget(self.window)
        self.main_layout = QVBoxLayout()
        self.window.setLayout(self.main_layout)

        ########  adding the player controls, slider and timer ########
        # Play and pause button
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setText("P")
        self.playing = False
        self.play_pause_btn.clicked.connect(self.play_pause)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)

        # adding slider and player to a layout
        self.player_controls = QHBoxLayout()
        self.player_controls.addWidget(self.play_pause_btn)
        self.player_controls.addWidget(self.slider)

        self.main_layout.addLayout(self.player_controls)

        # setting up a timer to play the video automatically
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.timer_update)

        ########   adding annotation related code ########

        self.labels_view = QHBoxLayout()
        self.labeling = False

        view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
        view.setMaximumHeight(100)
        self.labels_view.addWidget(view)

        # max_x = self.max_frames
        self.w1 = view.addPlot()
        # w1.setLimits(yMin=0,yMax=3,minYRange=0,maxYRange=3)
        self.w1.setMouseEnabled(y=False)
        self.w1.setXRange(0, 50, padding=None)
        self.w1.setYRange(0, 3, padding=None)
        ax_b = self.w1.getAxis('bottom')  # This is the trick
        # ax_b.setTicks([])
        # ax_l = w1.getAxis('left')  # This is the trick
        # ax_l.setHeight(100)
        self.main_layout.addLayout(self.labels_view)

        self.s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))
        self.s1.sigClicked.connect(self.clicked)
        self.w1.addItem(self.s1)

        self.time_cursor = pg.InfiniteLine(movable=True, angle=90)
        self.time_cursor.sigDragged.connect(self.dragged)
        self.w1.addItem(self.time_cursor)

        ## Make all plots clickable
        self.lastClicked = []
        self.labels = []
        self.loaded_labels = None

        import_export_view = QVBoxLayout()

        self.labeling_l = QLabel()
        self.labeling_l.setText("")
        import_export_view.addWidget(self.labeling_l)

        self.current_index = QLabel()
        self.current_index.setText("Frame:0")
        import_export_view.addWidget(self.current_index)

        import_btn = QPushButton()
        import_btn.setText("import")
        import_btn.clicked.connect(self.import_labels)
        import_export_view.addWidget(import_btn)

        export = QPushButton()
        export.setText("export")
        export.clicked.connect(self.export_labels)
        import_export_view.addWidget(export)

        self.labels_view.addLayout(import_export_view)

        data_paths = None
        self.label_path = None

        ######## User defined sensor layout #######
        if args.i.endswith(".yaml"):
            with open(args.i) as f:
                settings = yaml.load(f, Loader=yaml.FullLoader)
                print(settings)
                data_paths = settings["data_paths"]
                self.label_path = settings["label_path"]

        else:
            settings = {"data_paths": [args.i]}

        print(args.l + "." + args.l + "(settings)")
        cmd_str = args.l + "." + args.l + "(settings)"
        self.sensor_layout = eval(cmd_str)
        self.max_frames = self.sensor_layout.get_len() - 1
        self.main_layout.addLayout(self.sensor_layout)

        print(self.max_frames)
        self.slider.setRange(0, self.max_frames)

        self.time = self.sensor_layout.get_time()
        self.time = pd.DataFrame(self.time)

    def slider_changed(self, frame):
        self.frame = frame
        self.w1.setXRange(self.frame - 20, self.frame + 20, padding=None)
        self.time_cursor.setValue(self.frame)
        #self.current_index.setText("Frame:" + str(frame))
        timestamp = self.time.iloc[frame][0]
        timestamp = pd.to_datetime(timestamp,unit='ms')
        self.current_index.setText("Frame:" + timestamp.strftime('%Y-%m-%d %X'))



        self.update_components()

    def timer_update(self):
        self.frame = self.frame + 1
        self.update_slider(self.frame)

    def update_slider(self, frame):
        self.slider.setValue(frame)

    def play_pause(self):
        if self.playing:
            self.update_timer.stop()
        else:
            self.update_timer.start(100)
        self.playing = ~self.playing

    def keyPressEvent(self, e):
        if e.key() == 65:
            # Moving left
            self.update_slider(self.frame - 1)
        elif e.key() == 68:
            # Moving right
            self.update_slider(self.frame + 1)

        if e.key() == 83:
            self.labeling = ~self.labeling
            if self.labeling:
                self.labeling_l.setText("Labeling")
            else:
                self.labeling_l.setText("")

        if self.labeling or e.key() == 87:
            # Labeling butto n pressed
            if self.frame not in self.labels:
                self.labels += [self.frame]
                spots = [{'pos': [self.frame, 1], 'data': 1}]
                self.s1.addPoints(spots)
            else:
                self.labels.remove(self.frame)
                self.repaint_labels()

    def clicked(self, plot, points):
        self.lastClicked
        for p in self.lastClicked:
            p.resetPen()
        self.update_slider(points[-1].pos()[0])
        for p in points:
            p.setPen('b', width=2)
        self.lastClicked = points

    def repaint_labels(self):
        spots = [{'pos': [l, 1], 'data': 1} for l in self.labels]
        self.s1.clear()
        self.s1.addPoints(spots)

    def dragged(self, e):
        e.setValue(np.ceil(e.value()))
        self.update_slider(e.value())

    def import_labels(self):
        if self.label_path != None:
            print(self.label_path)
            self.loaded_labels = pd.read_csv(self.label_path, index_col=0)
            self.labels = self.loaded_labels.copy()
            self.labels = self.labels.dropna()
            self.labels = self.labels.index.values.tolist()
            self.repaint_labels()
        else:
            print("still working on this feature. In the meanwhile you can specify it using the yaml settings file")

    def export_labels(self):

        output_file = self.label_path

        if os.path.exists(output_file):
            save_timestamp = str(int(datetime.now().timestamp()))
            print("Label file exist. Creating another one", save_timestamp)
            output_file = output_file[:-4] + save_timestamp + ".csv"


        self.time.iloc[self.labels].to_csv(output_file, header=True)

        # print(self.loaded_labels)
        # print(self.labels)
        # if self.loaded_labels is None:
        #     self.time.iloc[self.labels].to_csv(output_file, header=True)
        # else:
        #     self.loaded_labels["updated"] = None
        #     self.loaded_labels.loc[self.labels, "updated"] = 1
        #     self.loaded_labels.to_csv(output_file, header=True)

    def update_components(self):
        self.sensor_layout.update_components(self.frame)


if __name__ == '__main__':
    app = QApplication([])

    args = parser.parse_args()

    print(args)

    window = SenseView(args)
    window.show()
    app.exit(app.exec_())
