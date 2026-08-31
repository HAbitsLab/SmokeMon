import serial
import numpy as np
import pandas as pd
import cv2
import os
import time
import glob
import matplotlib.pyplot as plt




#getting the usb serial port. This might be specific to MacOS and linux
ports = os.listdir("/dev/")
ports = [p for p in ports if p.startswith("cu.usb")]
# file_name = "D1_far.csv"

if len(ports) == 0:
    print ("Device not found")
elif len(ports) > 1:
    print("Too many usb devices are connected. currently assuming that there is one usb device only.")

ser = serial.Serial(port=os.path.join("/dev/",ports[0]),baudrate=115200,timeout=1)
# ser1 = serial.Serial(port=os.path.join("/dev/",ports[1]),baudrate=115200,timeout=1)
# ser2 = serial.Serial(port=os.path.join("/dev/",ports[2]),baudrate=115200,timeout=1)



init_flag = 0
res= 1
show_image = True
max_val = []
Ta_val = []
# i = 0
prev_mills = 0
mills = 0
while True:

    if (ser.inWaiting() > 0):
        line = ser.readline()
        # line1 = ser1.readline()
        # line2 = ser2.readline()
        try:
            line = line.decode("utf-8")
            # line1 = line1.decode("utf-8")
            # line2 = line2.decode("utf-8")
        except:
            continue
        line_arr = line.split(',')
        # line_arr1 = line1.split(',')
        # line_arr2 = line2.split(',')

        data = line_arr[:-1]
        # data1 = line_arr1[:-1]
        # data2 = line_arr2[:-1]

        # mills = int(data[1]) - prev_mills
        # prev_mills = int(data[1])



        #sensorData = np.array(data[2:]).astype(float)
        sensorData = np.array(data).astype(float)
        # sensorData1 = np.array(data1).astype(float)
        # sensorData2 = np.array(data2).astype(float)

        # output = "%d,%0.2f,%0.2f,%0.2f\n" %(time.time()*1000,sensorData.max(),sensorData1.max(),sensorData2.max())
        # f = open(file_name, "a")
        # print(output)
        # f.write(output)
        # f.close()
        # print(sensorData.max())
        # head = sensorData[512:-5]
        # # sensorData[512:] *= 0
        # human = sensorData[720] - 0.5
        # head = head[(head>=human)&(head<39)]
        # head = head.shape[0]/250
        # head = np.round(head*100,2)
        # print(head,human)

        # max_val += [sensorData.max()]
        print(sensorData.max(),sensorData.min(),sensorData.mean())
        # Ta_val += [data[-1]]
        if show_image:
            #Prepocessing the image to view as a heatmap
            sensorData = (sensorData * res)
            sensorData = sensorData - sensorData.min()
            sensorData = sensorData / sensorData.max()
            sensorData = sensorData*225
            sensorData = sensorData.astype(np.uint8)
            sensorData = sensorData.reshape(24,32)
            sensorData = cv2.resize(sensorData, (320,240))
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
        # means_pd.to_csv("means_second.csv")qqqqqq
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