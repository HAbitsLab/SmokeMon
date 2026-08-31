import os
import pandas as pd
import cv2
import numpy as np


class Camera:

    def __init__(self,timestamp=True):
        self.img_list = None
        self.timestamp = timestamp
        self.current_index = 0
        self.imgs_pd = pd.DataFrame(columns=["timestamp", "path"])

    def load_img_list(self, folder_path):
        img_list = os.listdir(folder_path)
        img_ext = img_list[0].split(".")[1]
        img_list = [int(img[:-(len(img_ext)+1)]) for img in img_list if (not img.startswith("."))]
        img_list.sort()


        # todo: add code to resample the image list
        self.current_index = min(img_list)

        self.imgs_pd = pd.DataFrame(data=img_list, columns=["timestamp"])
        self.imgs_pd["path"] = [os.path.join(folder_path, str(img) + "."+img_ext) for img in img_list]

        return self.imgs_pd

    def get_image(self,index):
        img_path = self.imgs_pd["path"].values[index]
        img = cv2.imread(img_path)
        img = np.rot90(img,-1)
        return img


    def get_time(self):
        return self.imgs_pd["timestamp"]





if __name__ == '__main__':
    main_dir = "../ExampleData/RGB/"
    video = Camera()
    imgs_pd = video.load_img_list(main_dir)
    img = video.get_image(0)
    print(img.shape)