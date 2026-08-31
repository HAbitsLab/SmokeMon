import random

import pandas as pd
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import sys


sys.path.insert(1, os.path.abspath(".."))
from base import BaseDataLoader
from .augs import *

# ------ normalizatioun parametrs excluding P11 --------
# DATAST_MEAN = 23.288445
# DATASET_STD = 9.973674
# --------------------------------------


DATAST_MEAN = 24.863447
DATASET_STD = 8.71266

#"vertical_flip": 0.2,
# "blur": 0.2,
# "addcig": 0.2,
# "remove_cig_2": 0.2,
# "jitter_brightness": 0.2,
# "random_crop": 0.3,
# "shuffle": 0,
# "random_rot": 20
class SmokeMonDataLoader(BaseDataLoader):

    def __init__(self, data_dir, p_list, batch_size=64, label_type="label", shuffle=False, validation_split=0.0,
                 num_workers=1, training=True, balance_ratio=0, GT_sessions = False, temporal_window = 0, confounding=False ,augs = None):
        if training:
            trsfm = transforms.Compose([
                Blur(augs['blur']),
                #Random_Temporal_Shuffle(0.2),
                Remove_Cigg_2(augs['remove_cig_2']),
                Add_Cigg(augs['addcig']),
                Jitter_Brightness(augs['jitter_brightness']),
                Random_Crop(augs['random_crop']),
                Random_Flip(augs['vertical_flip']),
                #Remove_Cigg(0.2),
                #Add_Cigg(0.2),
                #transforms.Resize(28),
                #transforms.RandomResizedCrop(28),
                transforms.ToTensor(),
                #transforms.RandomVerticalFlip(augs['vertical_flip']),
                #transforms.RandomHorizontalFlip(augs['vertical_flip']),
                #transforms.Normalize((DATAST_MEAN,), (DATASET_STD,)),
                # transforms.RandomResizedCrop((28,37)),
                transforms.RandomRotation(degrees=augs['random_rot']),
                #transforms.RandomErasing(value=27)
            ])
        else:
            trsfm = transforms.Compose([
                #transforms.Resize(28),
                #transforms.RandomResizedCrop(28),
                transforms.ToTensor(),
                #transforms.Normalize((DATAST_MEAN,), (DATASET_STD,))
                # transforms.RandomResizedCrop((28,37)),
                # transforms.RandomRotation(degrees=20),
                # transforms.RandomVerticalFlip(),
            ])

        self.data_dir = data_dir
        # self.dataset = SmokeMonInWildDataset(self.data_dir, p_list, label_type, training, transform=trsfm,
        #                                    balance_ratio=balance_ratio, GT_sessions=GT_sessions,temporal_window= temporal_window, confounding=confounding)

        self.dataset = SmokeMonInWildDataset_SlowFast(self.data_dir, p_list, label_type, training, transform=trsfm,
                                            balance_ratio=balance_ratio, GT_sessions=GT_sessions,temporal_window= temporal_window, confounding=confounding)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class SmokeMonInWildDataset_SlowFast(Dataset):
    def __init__(self, data_dir, p_list, label_type="label", training=True, transform=None, balance_ratio=0.3,
                 GT_sessions=False, temporal_window=0, confounding=False):
        print("loading data for:", p_list)
        self.temporal_window = temporal_window
        self.balance_ratio = balance_ratio
        self.transform = transform
        self.label_type = label_type
        all_data = []
        all_labels = []
        all_puff_summary = []

        #included_list = ["P11","P9","P10","P8","P7","P6"]
        included_list = []
        for p in included_list:
            data_path = os.path.join(data_dir, "Controlled", p, "sessions", "all_session_data.feather")
            labels_path = os.path.join(data_dir, "Controlled", p, "sessions", "all_sessions_labels.feather")

            #data_path = os.path.join(data_dir, p, "all_data.feather")
            #labels_path = os.path.join(data_dir, p, "all_labels.feather")
            data = pd.read_feather(data_path)
            labels = pd.read_feather(labels_path)
            puff_summary = pd.read_csv(os.path.join(data_dir, "Controlled", p, "puff_summary.csv"), index_col=0)

            labels["p"] = p
            puff_summary["p"] = p
            all_puff_summary += [puff_summary]
            all_labels += [labels]
            all_data += [data]



        for p in p_list:
            #data_path = os.path.join(data_dir, "In-Wild", p, "sessions", "all_session_data.feather")
            #labels_path = os.path.join(data_dir, "In-Wild", p, "sessions", "all_sessions_labels.feather")
            data_path = os.path.join(data_dir, "In-Wild", p, "all_data.feather")
            labels_path = os.path.join(data_dir, "In-Wild", p, "all_labels.feather")

            #data_path = os.path.join(data_dir, p, "all_data.feather")
            #labels_path = os.path.join(data_dir, p, "all_labels.feather")
            data = pd.read_feather(data_path)
            data = data.fillna(0)
            labels = pd.read_feather(labels_path)
            puff_summary = pd.read_csv(os.path.join(data_dir, "In-Wild", p, "puff_summary.csv"), index_col=0)


            if GT_sessions:
                for s in labels.session.unique():
                    if s == 0:
                        continue
                    s_index = labels[labels.session == s].index
                    labels.loc[s_index[0]:s_index[-1], "session"] = s

                labels = labels[labels.session != 0]
                data = data.loc[labels.index]

            if confounding:
                print("loading confounding data for:", p)
                con_data_path = os.path.join(data_dir, p, "confounding", "confounding_data.feather")
                con_labels_path = os.path.join(data_dir, p, "confounding", "confounding_labels.feather")
                if os.path.exists(con_data_path) == True:
                    conf_data = pd.read_feather(con_data_path)
                    conf_data = conf_data.fillna(0)
                    conf_labels = pd.read_feather(con_labels_path)
                    idxs = conf_labels[conf_labels.label == 0].index.values
                    np.random.shuffle(idxs)
                    idxs = idxs[:1000]
                    conf_labels["p"] = p
                    conf_data = conf_data.loc[idxs]
                    conf_labels = conf_labels.loc[idxs]
                    all_labels += [conf_labels]
                    all_data += [conf_data]


            if training and self.balance_ratio != 0:
                # balancing the dataset
                pos = labels[labels.label == 1].index.values
                neg = labels[labels.label == 0].index.values
                true_ratio = pos.shape[0] / neg.shape[0]
                neg_number = int(pos.shape[0] / self.balance_ratio)
                np.random.shuffle(neg)
                neg = neg[:neg_number]  # sampling from neg indexes to match balance ratio
                balanced = np.append(pos, neg)
                np.random.shuffle(balanced)

                print(f"Loading Participant {p} data: True class ratio: {true_ratio}, balanced ratio: {pos.shape[0] / neg.shape[0]}")

                data = data.loc[balanced]
                labels = labels.loc[balanced]

            labels["p"] = p
            puff_summary["p"] = p
            all_puff_summary += [puff_summary]
            all_labels += [labels]
            all_data += [data]

        self.all_data = pd.concat(all_data, sort=False, ignore_index=True)
        self.all_labels = pd.concat(all_labels, sort=False, ignore_index=True)
        self.all_puff_summary = pd.concat(all_puff_summary, sort=False, ignore_index=True)

    def __len__(self):
        return self.all_labels.shape[0]

    def __getitem__(self, i):
        if self.temporal_window != 0:
            indices_fast = np.arange(i-self.temporal_window,i+self.temporal_window+1, 1)
            indices_fast[indices_fast < 1] = 0
            indices_fast[indices_fast >= self.all_labels.shape[0]] = self.all_labels.shape[0] - 1

            data_fast = self.all_data.iloc[indices_fast].values

            diff = np.diff(data_fast[:, 0])

            gap = np.where(diff > 5000)[0]

            if gap.shape[0]>0:
                remove_from_after = np.max(gap) + 1
                indices_fast[remove_from_after:] = indices_fast[remove_from_after - 1]
                data_fast = self.all_data.iloc[indices_fast].values



            data_fast = data_fast[:, 1:-1].reshape((self.temporal_window*2+1, 24, 32)).astype('float32')







            #indices_slow = np.arange(i - self.temporal_window*50, i + self.temporal_window*50 + 1, 20)
            indices_slow = indices_fast[::3]
            indices_slow[indices_slow < 1] = 0
            indices_slow[indices_slow >= self.all_labels.shape[0]] = self.all_labels.shape[0] - 1

            data_slow = self.all_data.iloc[indices_slow].values

            diff = np.diff(data_slow[:, 0])

            gap = np.where(diff > 5000)[0]

            if gap.shape[0] > 0:
                remove_from_after = np.max(gap) + 1
                indices_slow[remove_from_after:] = indices_slow[remove_from_after - 1]
                data_slow = self.all_data.iloc[indices_slow].values

            data_slow = data_slow[:, 1:-1].reshape((int((self.temporal_window*2+1)/3), 24, 32)).astype('float32')


            #[1:].reshape(24, 32).astype('float32').copy()
            #data = np.dstack((data, data, data)).astype('float32')
            #data = (data - DATAST_MEAN) / DATASET_STD
            #data = (255 * (data - np.min(data)) / np.ptp(data)).astype(np.uint8)

            data_fast = data_fast.transpose((1,2,0))
            data_slow = data_slow.transpose((1, 2, 0))
            #data = np.expand_dims(data, axis=1)
            pass
        else:
            pass
            # data = self.all_data.iloc[i].values[1:].reshape(24, 32).astype('float32').copy()
            # # data = (data - DATAST_MEAN)/DATASET_STD
            # # thresh = threshold_otsu(data)
            # #data[data>40] = 40
            # data = Image.fromarray(data)

        #print(data.shape)
        #data = Image.fromarray(data)
        #data = data.convert('RGB')
        label = self.all_labels[self.label_type].iloc[indices_fast[indices_fast.shape[0]//2]].astype('int')
        b_label = self.all_labels["cigg"].iloc[indices_fast[indices_fast.shape[0]//2]].astype('int')
        #b_label = 1 if np.max(data[indices.shape[0]//2]) > 50 else 0
        #b_label = 0
        p_timestamp = self.all_labels.iloc[indices_fast[indices_fast.shape[0]//2]][["p", "0"]].values.tolist()

        if self.transform:
            data_fast = self.transform(data_fast)
            data_slow = self.transform(data_slow)
            #data_fast = torch.tensor(data_fast)
            #data_slow = torch.tensor(data_slow)



        # if label == 1:
        #     label = np.array([1,0], dtype = int)
        # else:
        #     label = np.array([0, 1], dtype = int)
        data_fast = torch.unsqueeze(data_fast,0)
        data_slow = torch.unsqueeze(data_slow,0)
        data = torch.concat((data_fast, data_slow), dim=1)

        return data, label, b_label, p_timestamp


class SmokeMonInWildDataset(Dataset):
    def __init__(self, data_dir, p_list, label_type="label", training=True, transform=None, balance_ratio=0.3,
                 GT_sessions=False, temporal_window=0, confounding=False):
        print("loading data for:", p_list)
        self.temporal_window = temporal_window
        self.balance_ratio = balance_ratio
        self.transform = transform
        self.label_type = label_type
        all_data = []
        all_labels = []
        all_puff_summary = []

        #included_list = ["P11","P9","P10","P8","P7","P6"]
        included_list = []
        for p in included_list:
            data_path = os.path.join(data_dir, "Controlled", p, "sessions", "all_session_data.feather")
            labels_path = os.path.join(data_dir, "Controlled", p, "sessions", "all_sessions_labels.feather")

            #data_path = os.path.join(data_dir, p, "all_data.feather")
            #labels_path = os.path.join(data_dir, p, "all_labels.feather")
            data = pd.read_feather(data_path)
            labels = pd.read_feather(labels_path)
            puff_summary = pd.read_csv(os.path.join(data_dir, "Controlled", p, "puff_summary.csv"), index_col=0)

            labels["p"] = p
            puff_summary["p"] = p
            all_puff_summary += [puff_summary]
            all_labels += [labels]
            all_data += [data]



        for p in p_list:
            #data_path = os.path.join(data_dir, "In-Wild", p, "sessions", "all_session_data.feather")
            #labels_path = os.path.join(data_dir, "In-Wild", p, "sessions", "all_sessions_labels.feather")
            data_path = os.path.join(data_dir, "In-Wild", p, "all_data.feather")
            labels_path = os.path.join(data_dir, "In-Wild", p, "all_labels.feather")

            #data_path = os.path.join(data_dir, p, "all_data.feather")
            #labels_path = os.path.join(data_dir, p, "all_labels.feather")
            data = pd.read_feather(data_path)
            data = data.fillna(0)
            labels = pd.read_feather(labels_path)
            puff_summary = pd.read_csv(os.path.join(data_dir, "In-Wild", p, "puff_summary.csv"), index_col=0)


            if GT_sessions:
                for s in labels.session.unique():
                    if s == 0:
                        continue
                    s_index = labels[labels.session == s].index
                    labels.loc[s_index[0]:s_index[-1], "session"] = s

                labels = labels[labels.session != 0]
                data = data.loc[labels.index]

            if confounding:
                print("loading confounding data for:", p)
                con_data_path = os.path.join(data_dir, p, "confounding", "confounding_data.feather")
                con_labels_path = os.path.join(data_dir, p, "confounding", "confounding_labels.feather")
                if os.path.exists(con_data_path) == True:
                    conf_data = pd.read_feather(con_data_path)
                    conf_data = conf_data.fillna(0)
                    conf_labels = pd.read_feather(con_labels_path)
                    idxs = conf_labels[conf_labels.label == 0].index.values
                    np.random.shuffle(idxs)
                    idxs = idxs[:1000]
                    conf_labels["p"] = p
                    conf_data = conf_data.loc[idxs]
                    conf_labels = conf_labels.loc[idxs]
                    all_labels += [conf_labels]
                    all_data += [conf_data]


            if training and self.balance_ratio != 0:
                # balancing the dataset
                pos = labels[labels.label == 1].index.values
                neg = labels[labels.label == 0].index.values
                true_ratio = pos.shape[0] / neg.shape[0]
                neg_number = int(pos.shape[0] / self.balance_ratio)
                np.random.shuffle(neg)
                neg = neg[:neg_number]  # sampling from neg indexes to match balance ratio
                balanced = np.append(pos, neg)
                np.random.shuffle(balanced)

                print(
                    f"Loading Participant {p} data: True class ratio: {true_ratio}, balanced ratio: {pos.shape[0] / neg.shape[0]}")

                data = data.loc[balanced]
                labels = labels.loc[balanced]

            labels["p"] = p
            puff_summary["p"] = p
            all_puff_summary += [puff_summary]
            all_labels += [labels]
            all_data += [data]

        self.all_data = pd.concat(all_data, sort=False, ignore_index=True)
        self.all_labels = pd.concat(all_labels, sort=False, ignore_index=True)
        self.all_puff_summary = pd.concat(all_puff_summary, sort=False, ignore_index=True)

    def __len__(self):
        return self.all_labels.shape[0]

    def __getitem__(self, i):
        if self.temporal_window != 0:
            indices = np.arange(i-self.temporal_window,i+self.temporal_window+1, 1)
            indices[indices < 1] = 0
            indices[indices >= self.all_labels.shape[0]] = self.all_labels.shape[0] - 1

            data = self.all_data.iloc[indices].values

            diff = np.diff(data[:, 0])

            gap = np.where(diff > 5000)[0]

            if gap.shape[0]>0:
                remove_from_after = np.max(gap) + 1
                indices[remove_from_after:] = indices[remove_from_after - 1]
                data = self.all_data.iloc[indices].values



            data = data[:, 1:-1].reshape((self.temporal_window*2+1, 24, 32)).astype('float32')


            #[1:].reshape(24, 32).astype('float32').copy()
            #data = np.dstack((data, data, data)).astype('float32')
            #data = (data - DATAST_MEAN) / DATASET_STD
            #data = (255 * (data - np.min(data)) / np.ptp(data)).astype(np.uint8)

            data = data.transpose((1,2,0))
            #data = np.expand_dims(data, axis=1)
            pass
        else:
            data = self.all_data.iloc[i].values[1:].reshape(24, 32).astype('float32').copy()
            # data = (data - DATAST_MEAN)/DATASET_STD
            # thresh = threshold_otsu(data)
            #data[data>40] = 40
            data = Image.fromarray(data)

        #print(data.shape)
        #data = Image.fromarray(data)
        #data = data.convert('RGB')
        label = self.all_labels[self.label_type].iloc[indices[indices.shape[0]//2]].astype('int')
        b_label = self.all_labels["cigg"].iloc[indices[indices.shape[0]//2]].astype('int')
        #b_label = 1 if np.max(data[indices.shape[0]//2]) > 50 else 0
        #b_label = 0
        p_timestamp = self.all_labels.iloc[indices[indices.shape[0]//2]][["p", "0"]].values.tolist()

        if self.transform:
            data = self.transform(data)



        # if label == 1:
        #     label = np.array([1,0], dtype = int)
        # else:
        #     label = np.array([0, 1], dtype = int)
        data = torch.unsqueeze(data,0)
        return data, label, b_label, p_timestamp

if __name__ == '__main__':
    #in_wild_dir = "/ssd2/thermo-smoking/Data/In-Wild/"
    pilot = "/ssd2/thermo-smoking/Data/"
    #in_wild_dir = "../../../Data/In-Wild/"
    # p_list = ["P11","P10","P9","P8","P6"]
    p_list = ["P11"]
    trsfm = transforms.Compose([
        #Random_Temporal_Shuffle(1),
        #Blur(1),
        #Random_V_Flip(1),

        #Random_Crop(1),
        #Add_Cigg(1),
        #Remove_Cigg_2(1),
        #Jitter_Brightness(1),
        #Remove_Cigg(1),
        #transforms.Resize(28),
        # transforms.RandomResizedCrop(28),
        transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(1)
        #transforms.Normalize((DATAST_MEAN,), (DATASET_STD,)),
        # transforms.RandomResizedCrop((28,37)),
        #transforms.RandomRotation(degrees=20),
        #transforms.RandomAffine(20)
        #transforms.RandomVerticalFlip(),
        #transforms.RandomHorizontalFlip()
    ])
    dataset = SmokeMonInWildDataset(pilot, p_list, transform=trsfm, training=False, GT_sessions=True, temporal_window=5, confounding=False)
    #data_loader = SmokeMonDataLoader(in_wild_dir, p_list, batch_size=1, confounding=True)

    #results = pd.read_csv()
    # data_loader = SmokeMonDataLoader(in_wild_dir, p_list, batch_size=1)
    fig = plt.figure(figsize=(15, 15))
    columns = 5
    rows = 2
    for x in range(4161,18000):
        label = dataset.__getitem__(x)[1]
        if label == 0:
            img = dataset.__getitem__(x)[0]
            img = img[0].permute(1,2,0)
            # for i in range(1, columns * rows + 1):
            #     fig.add_subplot(rows, columns, i)
            #     alpha = 1  # Contrast control (1.0-3.0)
            #     beta = 0  # Brightness control (0-100)
            #
            #     img[:,:,i-1] = (255 * (img[:,:,i-1] - np.min(img[:,:,i-1])) / np.ptp(img[:,:,i-1]))
            #
            #     #adjusted = cv2.convertScaleAbs(img[:,:,i-1], alpha=alpha, beta=beta)
            #
            #     im_show = cv2.applyColorMap(img[:,:,i-1].astype(np.uint8), cv2.COLORMAP_TURBO)
            #
            #     im_show = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
            #     plt.imshow(im_show)


            stacked_img1 = np.concatenate((img[:,:,0],img[:,:,1],img[:,:,2], img[:,:,3], img[:,:,4]), axis = 1)
            stacked_img2 = np.concatenate((img[:, :, 5], img[:, :, 6], img[:, :, 7], img[:, :, 8], img[:, :, 9]), axis=1)
            img = np.concatenate((stacked_img1, stacked_img2))

            #img = (255 * (img - np.min(img)) / np.ptp(img))
            #img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            #img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img, cmap="jet", vmin=0, vmax=50)
            #plt.savefig("test.png", bbox_inches='tight')
            break
            # cv2.imshow("demo", img)
            # cv2.waitKey()


    plt.show()

    #print(np.asarray(img))


