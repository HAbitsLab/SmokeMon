""" Wildcam1 sensor class
Wildcam1 has a camera and an GridEye
Sample Data collected from firmware vision X
Contact: Rawan
"""

import cv2
import matplotlib.pyplot as plt

from Sensors.Camera import Camera
from Sensors.Grideye import Grideye

SENSOR_POSITION = (50,105)
SENSOR_SIZE = (110,110)


class Wildcam1:

    def __init__(self, image_folder, csv_path):
        """
       :param image_folder: a string containing the path to the folder that has the RGB images
       :param csv_path: a string containing the path to the thermal csv file
       """

        # creating and initializing a Camera object
        self.camera = Camera()
        self.imgs_pd = self.camera.load_img_list(image_folder)

        # creating and initializing an GridEye object
        self.grideye = Grideye(csv_path)

    def get_data(self, index):
        """ Retrieve Wildcam1 data by index
       :param index: frame index
       :return: RGB frame array (320x240x3), thermal frame array (8x8)
       """
        img = self.camera.get_image(index)
        mlx = self.grideye.get_image(index)
        return img, mlx

    def get_time(self):
        """ Get the column that contains the timestamp
       :return: Dataframe containing 1 column that represent time in milliseconds
       """
        return self.imgs_pd["timestamp"]


    def get_overlay(self, index):
        """ Register the Grideye frame to the image to get an overlay mask
        :param index: frame index
        :return: Gray frame array (320x240) with a human mask overlayed on top
        """
        # get the RGB frame
        rgb = self.camera.get_image(index)

        # get the human mask
        thermal_overlay = self.grideye.get_human_simple(index)
        # thermal_overlay ^= 1

        # up sampling and resizing the thermal
        for i in range(5):
            thermal_overlay = cv2.pyrUp(thermal_overlay.astype('float32'))
        thermal_overlay = cv2.resize(thermal_overlay,SENSOR_SIZE)

        # Get the coordinated for the new position
        x = [SENSOR_POSITION[0], SENSOR_POSITION[0] + SENSOR_SIZE[0]]
        y = [SENSOR_POSITION[1], SENSOR_POSITION[1] + SENSOR_SIZE[1]]

        # convert to Gray scale and then register
        overlayed = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        overlayed[y[0]:y[1], x[0]:x[1]] = overlayed[y[0]:y[1], x[0]:x[1]] * thermal_overlay

        return overlayed


if __name__ == '__main__':
    image_folder = "../../sample_data/ 12"
    csv_path = "../../sample_data/12.csv"

    wildcam = Wildcam1(image_folder, csv_path)

    i = 1

    img, grideye = wildcam.get_data(i)
    plt.imshow(grideye)
    plt.show()

    plt.imshow(img)
    plt.show()

    plt.imshow(wildcam.get_overlay(i))
    plt.show()
