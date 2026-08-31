import pandas as pd
import os
import numpy as np

csv_path = "/Volumes/SMOKEMON/1632939.csv"
recorded_time = -1 # data recording time in minutes
mills_check = True

# FPS
data = pd.read_csv(csv_path,header = None)
if recorded_time > 0:
    fps = data.shape[0]/(recorded_time*60)
    print("fps:", fps)
else:
    if mills_check:
        mills = data[1].values
    else:
        mills = data[0].values
    fps = mills[1:] - mills[:-1]
    fps = 1000/np.median(fps)
    print("fps:",fps)

# File Size
size = os.path.getsize(csv_path)
print("File size (MB) :", size/1000000)
