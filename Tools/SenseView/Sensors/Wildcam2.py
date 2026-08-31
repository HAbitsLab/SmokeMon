import os
import pandas as pd
import cv2
import numpy as np

from Sensors.Camera import Camera
from Sensors.MLX import MLX
import matplotlib.pyplot as plt


def calibrate(sensor_size, sensor_position):
    x = [sensor_position[0], sensor_position[0] + sensor_size[0]]
    y = [sensor_position[1], sensor_position[1] + sensor_size[1]]
    return x, y

class Wildcam2:

    def __init__(self,image_folder,csv_path):

        self.camera = Camera()
        self.imgs_pd = self.camera.load_img_list(image_folder)

        self.mlx = MLX(csv_path)

    def get_data(self,index):
        img = self.camera.get_image(index)
        mlx = self.mlx.get_image(index)
        return img, mlx

    def get_time(self):
        return self.imgs_pd["timestamp"]

    def get_human(self, index):
        thresh = 27
        mlx = self.mlx.get_image(index)
        mlx[mlx < thresh] = 0
        mlx[mlx > 0] = 1
        return mlx


    def get_overlay(self, index):

        rgb = self.camera.get_image(index)

        thermal_overlay = self.get_human(index)
        sensor_size = [thermal_overlay.shape[1]*5, thermal_overlay.shape[0]*5]
        thermal_overlay = cv2.resize(thermal_overlay, (sensor_size[0], sensor_size[1]))

        x, y = calibrate(sensor_size, [70,70])

        overlayed = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        overlayed[y[0]:y[1], x[0]:x[1]] = overlayed[y[0]:y[1], x[0]:x[1]] * thermal_overlay

        return overlayed





if __name__ == '__main__':
    image_folder = "/Users/rma145/Desktop/SenseWhy_data/P0/Wild/Camera/Frame/2020_05_12/ 19"
    csv_path = "/Users/rma145/Desktop/SenseWhy_data/P0/Wild/Camera/Clean/2020-05-13.csv"
    wildcam = Wildcam2(image_folder,csv_path)

    i = 206

    img, mlx = wildcam.get_data(i)
    plt.imshow(mlx)
    plt.show()

    plt.imshow(img)
    plt.show()

    plt.imshow(wildcam.get_overlay(i))
    plt.show()


    # calibrate(mlx.shape, [0, 0])