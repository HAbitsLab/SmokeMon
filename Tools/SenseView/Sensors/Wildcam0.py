import os
import pandas as pd
import cv2
import numpy as np

from Sensors.Camera import Camera
from Sensors.Grideye import Grideye
import matplotlib.pyplot as plt

sensor_position = np.array([51, 90])
sensor_size = (170, 170)

def calibrate():

    x = [sensor_position[0], sensor_position[0] + sensor_size[0]]
    y = [sensor_position[1], sensor_position[1] + sensor_size[1]]
    return x, y


class Wildcam0:

    def __init__(self, image_folder, csv_path):
        self.camera = Camera()
        self.imgs_pd = self.camera.load_img_list(image_folder)

        self.grideye = Grideye(csv_path)

    def get_data(self, index):
        img = self.camera.get_image(index)
        mlx = self.grideye.get_image(index)
        return img, mlx

    def get_time(self):
        return self.imgs_pd["timestamp"]

    def get_human(self, index):
        human = self.grideye.get_human_2(index)
        return human

    def get_overlay(self, index):
        rgb = self.camera.get_image(index)

        thermal_overlay = self.get_human(index)
        # plt.imshow(thermal_overlay)
        # plt.show()
        # sensor_size = [thermal_overlay.shape[1]*5, thermal_overlay.shape[0]*5]
        for i in range(5):
            thermal_overlay = cv2.pyrUp(thermal_overlay.astype('float32'))
        thermal_overlay = cv2.resize(thermal_overlay,sensor_size)

        x, y = calibrate()

        overlayed = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        overlayed[y[0]:y[1], x[0]:x[1]] = overlayed[y[0]:y[1], x[0]:x[1]] * thermal_overlay

        return overlayed


if __name__ == '__main__':
    image_folder = "/Users/rma145/Desktop/WildCam/Data/P1/img/"
    csv_path = "/Users/rma145/Desktop/WildCam/Data/P1/gridEye.csv"
    wildcam = Wildcam0(image_folder, csv_path)

    i = 495

    img, grideye = wildcam.get_data(i)
    plt.imshow(grideye)
    plt.show()

    plt.imshow(img)
    plt.show()

    plt.imshow(wildcam.get_overlay(i))
    plt.show()

    # calibrate(mlx.shape, [0, 0])
