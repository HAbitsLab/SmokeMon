import os
import pandas as pd
import numpy as np
import cv2


class Video:

    def __init__(self, path=""):
        self.total_frames = 0
        self.fps = 0
        self.time = None
        if path != "":
            self.data = self.load_data(path)

    def load_data(self, folder_path,fps=None):
        self.data = cv2.VideoCapture(folder_path)
        if fps is None:
            fps = self.data.get(cv2.CAP_PROP_FPS)

        self.set_fps(fps)

        return self.data

    def set_fps(self, fps):
        self.fps = fps
        duration = self.data.get(cv2.CAP_PROP_FRAME_COUNT)/self.data.get(cv2.CAP_PROP_FPS)
        # duration = self.data.get(cv2.CAP_PROP_POS_MSEC)
        self.total_frames = int(duration * self.fps)
        self.time = np.arange(self.total_frames)
        self.data.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    def get_image(self, index):
        self.data.set(cv2.CAP_PROP_POS_MSEC, index * 1000/self.fps)
        ret, frame = self.data.read()
        frame = cv2.resize(frame, (320, 240))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.rotate(frame, 2)
        return frame

    def get_time(self):
        return self.time


if __name__ == '__main__':
    main_dir = "/Users/rma145/Desktop/SmokingData/P1/2020-02-28_15-18-47.mov"
    video = Video()
    video.load_data(main_dir,fps=4)
    print(video.fps)
    img = video.get_image(0)
    print(img.shape)
    print(video.total_frames)
    print(video.get_time())
