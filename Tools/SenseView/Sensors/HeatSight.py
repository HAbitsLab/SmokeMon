import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# todo: this should be moved to a util or preprocessing library
def quantize(sensor_quantized, quant_count, sensor_range):
    quant_width = sensor_range / quant_count
    quant_i = {}
    for i in range(quant_count):
        quant_i[i] = np.where(
            (sensor_quantized >= (quant_width * i)) & (sensor_quantized <= (quant_width * i + quant_width)))
        sensor_quantized[quant_i[i]] = ((i * quant_width) / sensor_range)
    return sensor_quantized


def quantize_human(sensor_quantized, human):
    quant_i = {}
    human_i = np.where((sensor_quantized >= human[0]) & (sensor_quantized <= human[1] + 3))
    colder = []

    sensor_range = human[0]
    quant_count = 10
    quant_width = sensor_range / quant_count

    for i in range(quant_count):
        quant_i[i] = np.where(
            (sensor_quantized >= (quant_width * i)) & (sensor_quantized <= (quant_width * i + quant_width)))
        sensor_quantized[quant_i[i]] = ((i * quant_width) / sensor_range) * 0.4

    off_set = human[1] + 3
    sensor_range = 80 - off_set
    quant_count = 20
    quant_width = sensor_range / quant_count

    for i in range(quant_count):
        quant_i[i] = np.where(
            (sensor_quantized >= (quant_width * i + off_set)) & (
                        sensor_quantized <= (quant_width * i + quant_width + off_set)))
        sensor_quantized[quant_i[i]] = ((i * quant_width) / sensor_range) * 0.4 + 0.6

    sensor_quantized[human_i] = 0.5

    return sensor_quantized


class HeatSight:
    def __init__(self, path=""):
        if path != "":
            self.data = self.load_csv(path)
        self.sensors_names = ["top", "right", "left", "bottom", "center"]
        self.transform = False

    def load_csv(self, path):
        self.data = pd.read_csv(path, header=None)

        # self.time = self.data[self.data.columns[0]]
        # self.frames = self.data.drop(columns=[0])

        self.time = self.data[self.data.columns[1]]
        self.frames = self.data.drop(columns=[0, 1])

        return self.data

    def get_image(self, index):

        calibrate = {"top": 0.13, "right": -2.29, "center": 0, "left": 1, "bottom": -0.73}
        try:
            frame = self.frames.iloc[index].values
        except:
            frame = self.frames.iloc[0].values * 0

        img = {}

        img["top"] = frame[0:64].reshape(8, 8)
        img["left"] = frame[64:64 + 64].reshape(8, 8)
        img["center"] = frame[64 + 64:64 + 64 * 2].reshape(8, 8)
        img["right"] = frame[64 + 64 * 2:64 + 64 * 3].reshape(8, 8)
        img["bottom"] = frame[64 + 64 * 3:64 + 64 * 4].reshape(8, 8)

        # print(img["top"])
        human_min_max = [img["top"][:, -1].mean() - 1, img["top"][:, -1].max()]
        # print(human_min_max)

        for name in self.sensors_names:
            img[name] = img[name] + calibrate[name]
            # tmp = img[name].mean() - img["center"].mean()
            # print("mean", name, img[name].mean(), tmp)

            img[name] = np.rot90(img[name])
            img[name][img[name] > 80] = 80
            img[name][img[name] < 0] = 0

            if self.transform:
                quant_count = 40
                # img[name] = quantize(np.copy(img[name]), quant_count=quant_count, sensor_range=80)
                img[name] = quantize_human(np.copy(img[name]), human_min_max)
            else:

                # img[name] = (img[name] - 21.58) / 2.55
                img[name] = img[name] / 80.0

        blank = np.zeros((8, 8))
        combined_top = np.hstack((blank, img["top"], blank))
        combined_center = np.hstack((img["right"], img["center"], img["left"]))
        combined_bottom = np.hstack((blank, img["bottom"], blank))
        combined = np.vstack((combined_top, combined_center, combined_bottom))
        combined = np.rot90(combined)
        return combined

    def get_time(self):
        return self.time


if __name__ == '__main__':
    csv_path = "/Users/rma145/Desktop/Octagon/Data/UIST/Jaya_home.CSV"
    HeatSight = HeatSight(csv_path)

    plt.imshow(HeatSight.get_image(0))
    plt.show()

    print(HeatSight.get_time())
