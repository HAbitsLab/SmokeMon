import serial
import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt


#getting the usb serial port. This might be specific to MacOS and linux
ports = os.listdir("/dev/")
ports = [p for p in ports if p.startswith("cu.usb")]

if len(ports) == 0:
    print ("Device not found")
elif len(ports) > 1:
    print("Too many usb devices are connected. currently assuming that there is one usb device only.")

ser = serial.Serial(port=os.path.join("/dev/",ports[0]),baudrate=115200,timeout=1)
show_image = True
res = 1
while True:

    if (ser.inWaiting() > 0):
        line = ser.readline()
        try:
            line = line.decode("utf-8")
        except:
            continue
        line_arr = line.split(',')

        data = line_arr[:-1]

        # mills = int(data[1]) - prev_mills
        # prev_mills = int(data[1])



        sensorData = np.array(data).astype(float)
        # print(sensorData.max())
        # max_val += [sensorData.max()]
        # Ta_val += [data[-1]]
        if show_image:
            #Prepocessing the image to view as a heatmap
            sensorData = (sensorData * res)
            sensorData = sensorData - sensorData.min()
            sensorData = sensorData / sensorData.max()
            sensorData = sensorData*225
            sensorData = sensorData.astype(np.uint8)
            sensorData = sensorData.reshape(8,8)
            sensorData = cv2.resize(sensorData, (200,200))
            sensorData = cv2.applyColorMap(sensorData, cv2.COLORMAP_JET)
            # print(sensorData)
            cv2.imshow("data", sensorData)
            # plt.imshow(sensorData)
            # plt.show()

            # #Saving the images
            # cv2.imwrite('images/{:>05}.png'.format(i), sensorData) #'images/{:>05}.png'.format(i), makes photo files

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        # means_pd = pd.DataFrame(data = means)
        # means_pd.to_csv("means_second.csv")
        break



# filename = 'video.avi'
# frames_per_second = 4.0
# res = '320p'
#
# VIDEO_TYPE = {
#     'avi': cv2.VideoWriter_fourcc(*'XVID'),
#     #'mp4': cv2.VideoWriter_fourcc(*'H264'),
#     'mp4': cv2.VideoWriter_fourcc(*'XVID'),
# }
#
# def get_video_type(filename):
#     filename, ext = os.path.splitext(filename)
#     if ext in VIDEO_TYPE:
#       return  VIDEO_TYPE[ext]
#     return VIDEO_TYPE['avi']
#
#
#
# out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_second, (320,240))
#
#
# image_list = glob.glob("images/*.png")
# sorted_images = sorted(image_list)
#
# clear_images = True
#
# for file in sorted_images:
#     print(file)
#     image_frame = cv2.imread(file, -1)
#     out.write(image_frame)
# for file in image_list:
#     os.remove(file)
#

# out.release()
cv2.destroyAllWindows()