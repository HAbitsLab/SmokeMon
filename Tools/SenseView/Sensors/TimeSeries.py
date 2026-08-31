import pandas as pd
import numpy as np


class TimeSeries:
    def __init__(self, path="", freq=0, off_set=0):
        self.off_set = off_set
        self.freq = freq
        if isinstance(path,str):
            self.data = self.load_data(path)
        else:
            self.data = path

    def load_data(self, path):

        self.data = pd.read_csv(path)
        # self.data[self.data.columns[0]] = pd.to_datetime(self.data.columns[0])
        return self.data

    def get_window(self, index, half_w=10):
        off_set_index = self.off_set + index
        win_s = off_set_index - half_w if off_set_index - half_w > 0 else 0
        win_e = off_set_index + half_w if off_set_index + half_w < self.data.shape[0] - 1 else self.data.shape[0] - 1

        win = self.data.iloc[win_s:win_e]

        return win

    def get_time(self):
        return self.data[self.data.columns[0]].values


if __name__ == '__main__':
    csv_path = '/Users/rma145/Desktop/thermo-smoking/Data/Pilot/Clean/P4/CP_ts.csv'
    ts_temp = pd.DataFrame(data=np.ones((100, 2)))
    ts = TimeSeries(ts_temp)

    i = 0

    win = ts.get_window(i)
    print(win.shape)
    # print(ts.data.values[:,2].max())
