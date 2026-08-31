import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops


def get_sensor_mask2(sensor_frame, sensor_size=(8, 8), flipped=True):
    # getting the threshold value
    thresh_mean = get_thresh(sensor_frame)
    vmin = thresh_mean - 1
    vmax = thresh_mean + 2

    # increase the resolution of the grideye
    i = 1
    while i <= 5:
        sensor_frame = cv2.pyrUp(sensor_frame)
        i = i + 1

    sensor_frame = cv2.resize(sensor_frame, (sensor_size[0], sensor_size[1]))
    if thresh_mean != 0:
        if (flipped):
            sensor_frame[sensor_frame <= vmin] = 0
            sensor_frame[sensor_frame > vmin] = 1
            sensor_frame[sensor_frame > vmax] = 0
        else:
            sensor_frame[sensor_frame <= vmin] = 1
            sensor_frame[sensor_frame > vmin] = 0
            sensor_frame[sensor_frame > vmax] = 1
    else:
        sensor_frame *= 0

    return sensor_frame


def get_thresh(sensor_frame):
    x = sensor_frame[(sensor_frame >= 21) & (sensor_frame <= 31)].flatten()

    if len(x) == 0:
        return 0

    # x_d = np.linspace(x.min(), x.max(), 1000)
    # density = sum(norm(xi).pdf(x_d) for xi in x)

    ret2, th2 = cv2.threshold(x.astype(np.uint8), 21, 31, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh_mean = x_d[np.argsort(density)[0]]
    thresh_mean = ret2
    return thresh_mean


class Grideye:
    def __init__(self, path):
        self.data = self.load_csv(path)

    def load_csv(self, path):
        self.data = pd.read_csv(path, header=None)

        return self.data

    def get_image(self, index):
        img = (self.data.iloc[index].values[1:]).astype(float)
        img = img.reshape(8, 8)
        return img

    def get_time(self):
        return self.data[self.data.columns[-1]]

    def get_human_simple(self, index):
        image = self.get_image(index)
        thresholds = threshold_multiotsu(image, 2)
        regions = np.digitize(image, bins=thresholds)
        return regions

    def get_human(self, index):
        """ Get a human mask
                Human pixels will have the value 1 and other pixels will be 0
               :param index: frame index
               :return: human mask array (8x8)
        """
        sensor_frame = self.get_image(index)

        # obtain a threshold using multiotsu method.
        # This should be return 2 values since we specify 3 as the number of histograms
        thresholds = threshold_multiotsu(sensor_frame, 3)
        regions = np.digitize(sensor_frame, bins=thresholds)  # assigning labels to each pixel based on the threshold
        print(regions)

        # we have 3 regions, the human region is always greater than or equal to 21
        human = np.where(thresholds >= 21)[0]

        if len(human) == 0:
            return sensor_frame * 0  # if no human range is detected return black frame
        else:
            human = human[0]  # Otherwise pick the index of the first threshold above 20.

        # creating a mask where the background is zero and the human is 1
        human_mask = regions * 0
        human_mask[(regions == human + 1)] = 1  # the region will be the index +1 since region start from 0

        # Check if there are more thresholds left. If yes, check if the  the
        if human + 1 < len(thresholds):
            if thresholds[human + 1] < 31:  # If yes, check if it is still in the human range (maximum human temp is 31)
                human_mask = regions * 0
                human_mask[(regions == human + 1) | (regions == human + 2)] = 1  # If yes, mark it as human

        return human_mask

    def get_diff_img(self, index):
        current = self.get_image(index)
        before = self.get_image(index - 1)
        diff = current - before

        current_h = self.get_human(index)
        before_h = self.get_human(index - 1)
        h = current_h + before_h
        h[h >= 1] = 1
        label_img = label(h)
        regions = regionprops(label_img)
        for props in regions:
            if props.coords[:, 0].min() == 0:
                diff[props.coords[:, 0], props.coords[:, 1]] = 0

        # diff[diff < 0] = 0
        # diff[diff > 0] = 1
        # return diff

        return diff

    def get_diff(self, index):
        diff = self.get_diff_img(index)
        # current = self.get_image(index)
        # before = self.get_image(index - 1)
        # diff = current - before
        diff_sum = np.abs(diff).sum()
        if diff_sum < 40:
            return 0
        else:
            return 1

    def get_diff_pd(self):
        diff = [0]
        for i in range(1, self.data.shape[0]):
            diff += [self.get_diff(i)]
        diff_pd = pd.DataFrame(data={"time": self.get_time().values,
                                     "diff": diff})

        return diff_pd


if __name__ == '__main__':
    csv_path = "../ExampleData/GridEye/gridEye.csv"
    grideye = Grideye(csv_path)

    plt.imshow(grideye.get_human(1))
    plt.show()

    # csv_path = "/Users/rma145/Desktop/Octagon/Data/CHI/CLEAN/P1/top.csv"
    # grideye = Grideye(csv_path)
    #
    # print(grideye.get_image(1))
    # plt.imshow(grideye.get_image(1))
    # plt.show()
