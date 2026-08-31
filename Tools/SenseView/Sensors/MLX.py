import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
import seaborn as sns
from scipy.spatial import distance
import math


def get_pre_head(img, session_head_center, session_head, human_range):
    # frame human
    human_region, core_region = get_human(img, human_range)

    # frame head
    head, head_features = get_head(core_region, session_head)

    return head, head_features


def get_human(img, human_range=[26, 35]):
    try:
        human = img[(img <= human_range[1]) & (img >= human_range[0])]
        thresholds = threshold_multiotsu(human, 3)
        regions = np.digitize(img, bins=thresholds)
    except:
        regions = img * 0

    human_region = regions.copy()
    human_region[human_region > 0] = 1

    core_region = regions.copy()
    core_region[core_region < 2] = 0

    return human_region, core_region


def get_head(core_region, session_head, prev_head=None, prev_head_features=None):
    head = label(core_region)
    blobs = regionprops(head)
    for props in blobs:
        if props.area < 15:
            head[head == props.label] = 0
            continue


        h_min_row, h_min_col, h_max_row, h_max_col = props.bbox
        # if max_row == 24 and max_col < 32 and min_col > 1:
        # if h_max_row == 24  and props.solidity > 0.7:
        if True:

            if prev_head is not None:
                min_row, min_col, max_row, max_col = prev_head_features["bbox"]

                if props.area > prev_head_features["area"]:
                    min_row, min_col, max_row, max_col = h_min_row, h_min_col, h_max_row, h_max_col

                frame_head_p = head.copy()
                frame_head_p[frame_head_p != props.label] = 0
                frame_head_p = frame_head_p[min_row:, min_col:max_col]
                prev_head_p = prev_head[min_row:, min_col:max_col]
                frame_head_p[frame_head_p > 1] = 1
                overlap = (prev_head_p & frame_head_p).sum() / (prev_head_p | frame_head_p).sum()
                if overlap > 0.4:
                    continue

            # min_row, min_col, max_row, max_col = props.bbox
            # session_head_p = session_head[-3:, min_col:max_col]
            # frame_head_p = head[-3:, min_col:max_col].copy()
            # frame_head_p[frame_head_p > 1] = 1
            # overlap = (session_head_p == frame_head_p).sum(axis=1).max() / frame_head_p.shape[1]
            #
            # if overlap > 0.6:
            #     continue

        # if dist > 5:
        head[head == props.label] = 0
        # continue

    if head.sum() == 0:
        if prev_head is not None:
            head = head + prev_head
        else:
            head = head + session_head
    head[head > 0] = 1

    head_blob = regionprops(head)[0]
    y0, x0 = head_blob.centroid
    head_orientation = head_blob.orientation
    x1 = x0 + math.cos(head_orientation) * 0.5 * head_blob.minor_axis_length
    y1 = y0 + math.sin(head_orientation) * 0.5 * head_blob.minor_axis_length
    y1 = 23 if y1 > 23 else y1
    x1 = 31 if x1 > 31 else x1

    head_center = (int(y0), int(x0))
    head_extreme = (head_blob.bbox[0], int(x0))

    head_features = {"orientation": head_orientation,
                     "bbox": head_blob.bbox,
                     "perimeter": head_blob.perimeter,
                     "extreme": head_extreme,
                     "centroid": head_center,
                     "area": head_blob.area
                     }
    head[head_extreme] = 5
    return head, head_features


def get_hands(head, head_features, human_region, session_region):
    head_rect = head.copy()
    # head_props = regionprops(head_rect)[0]
    minr, minc, maxr, maxc = head_features["bbox"]
    minc = minc - 1 if minc > 0 else 0
    head_rect[minr:, minc:maxc + 1] = 1

    hands = session_region.copy()
    hands += head_rect
    hands[hands > 0] = 1
    hands = hands ^ 1
    hands = human_region * hands

    hands = label(hands)
    blobs = regionprops(hands)
    hands_features = []
    for props in blobs:
        # min_row, min_col, max_row, max_col = props.bbox
        # if not (max_col == 32 or min_col == 0 or min_row == 0):
        #     hands[hands == props.label] = 0
        #     continue
        if props.area < 15:
            hands[hands == props.label] = 0
            continue

        y0, x0 = props.centroid
        hand_orientation = props.orientation

        hand_extreme_left = (int(y0), props.bbox[1] - 1)
        hand_extreme_right = (int(y0), props.bbox[3] - 1)
        hand_center = (int(y0), int(x0))

        hands_features += [{"dist_to_head_center": distance.euclidean(head_features["centroid"], props.centroid),
                            "dist_to_head_top_1": distance.euclidean(head_features["extreme"], hand_extreme_left),
                            "dist_to_head_top_2": distance.euclidean(head_features["extreme"], hand_extreme_right),
                            "orientation": hand_orientation,
                            "max_value": 0,
                            "min_value": 0,
                            "mean_value": 0,
                            "area": props.area,
                            "bbox": props.bbox,
                            "perimeter": props.perimeter,
                            "centroid": hand_center,
                            "extreme_minor": hand_extreme_left,
                            "extreme_major": hand_extreme_right,
                            }]

        hands[hand_extreme_left] = 5
        hands[hand_extreme_right] = 5

    return hands, hands_features


def get_human_parts(img, prev_img, session_region, session_head_center, session_head, human_range):
    # frame human
    human_region, core_region = get_human(img, human_range)

    prev_head, prev_head_features = get_pre_head(prev_img, session_head_center, session_head, human_range)

    # frame head
    head, head_features = get_head(core_region, session_head, prev_head, prev_head_features)

    # frame hands
    hands, hands_features = get_hands(head, head_features, human_region, session_region)
    hands = label(hands)
    hands[hands > 0] += 1
    return hands + head


class MLX:
    def __init__(self, path=""):
        if path != "":
            self.data = self.load_csv(path)

    def load_csv(self, path):
        self.data = pd.read_csv(path, header=None, error_bad_lines=False)
        if path.split("/")[-1] != "data.csv":
            self.data = self.data.drop(0, axis=0)
            self.data[769] = 0

        self.data = self.data.dropna(thresh=50)

        # remove this line after remving milliseconds
        # self.data[0] = self.data[0].apply(lambda d: d[:-3])

        # self.data[0] = pd.to_datetime(self.data[0], format='%Y-%m-%dT%H:%M:%S')

        try:
            self.session_img, self.session_regions = self.set_session_regions()
            self.cig, self.cig_center = self.get_cig_region()
            self.session_head_region, self.session_head_center = self.set_session_head()
            self.human_range = self.get_human_range()
        except:
            pass

        return self.data

    def get_colored(self, index):
        mlx = self.get_image(index)
        mlx[mlx > 70] = 70
        mlx = (mlx / 70) * 255
        # mlx = cv2.cvtColor(np.float32(mlx), cv2.COLOR_GRAY2BGR)
        # mlx = cv2.cvtColor(np.float32(mlx),cv2.COLOR_GRAY2RGB)
        mlx = cv2.applyColorMap(mlx.astype(np.uint8), cv2.COLORMAP_JET)
        mlx = cv2.bitwise_not(mlx)
        return (mlx)

    def get_image(self, index):
        print(self.data.iloc[index].values[1:1 + 24 * 32].shape)
        try:
            if self.data.shape[1] > 770:
                img = self.data.iloc[index].values[2:2 + 24 * 32].astype(float)
            elif self.data.shape[1] == 767:
                img = self.data.iloc[index].values[1:1 + 24 * 32].astype(float)
                img = np.append([img.mean()],img)
                img = np.append(img,[img.media()])
            else:
                img = self.data.iloc[index].values[1:1 + 24 * 32].astype(float)

            img = img.reshape(24, 32)
            img[img>40] = 40
            # img = np.flipud(img)
            # img = np.rot90(img, -1)
            # img = np.rot90(img)
        except:
            print("no image present")
            img = np.zeros((24, 32))

        return img

    def get_time(self):
        return self.data[0].values

    def get_human(self, i):
        img = self.get_image(i)
        if i != 0:
            prev_img = self.get_image(i - 1)
        else:
            prev_img = img.copy()

        # session head
        session_region = self.session_regions
        session_head_center = self.session_head_center
        session_head = self.session_head_region
        human_range = self.human_range

        human_parts = get_human_parts(img, prev_img, session_region, session_head_center, session_head, human_range)

        return human_parts

    def get_human_range(self):

        max_human = self.session_img[self.session_regions > 0].max()
        min_human = self.session_img[self.session_regions > 0].min() - 1.5
        # min_human = 29
        mean_human = self.session_img[self.session_head_region > 0].mean()

        # if max_human - mean_human < 3:
        #     min_human = mean_human - 1

        return [min_human, max_human]

    def get_cig_region(self):
        cig = self.data[self.data > 50]
        cig = cig.fillna(0)
        cig = cig.max().values[1:-1]
        cig[cig > 0] = 1
        cig = cig.reshape(24, 32)
        cig_center = regionprops(cig.astype(int))[0].centroid
        cig_center = [int(cig_center[0]), int(cig_center[1])]

        return cig, cig_center

    def set_session_regions(self):
        data = self.data[self.data.min(axis=1) > -30]
        img = data.mean().values[1:-1]
        thresholds = threshold_multiotsu(img, 4)
        regions = np.digitize(img, bins=thresholds)

        regions[regions == 1] = 0

        regions = regions.reshape(24, 32)
        img = img.reshape(24, 32)
        # img = np.flipud(img)
        # img = np.rot90(img, -1)

        return img, regions

    def set_session_head(self):
        session_head_region = self.session_regions.copy()
        session_head_extra = session_head_region.copy()
        # session_head_region[session_head_region != 3] = 0

        session_head = label(session_head_region)
        blobs = regionprops(session_head)
        session_head_center = [0, 0]

        for props in blobs:
            if props.area < 25:
                session_head[session_head == props.label] = 0
                continue

            y0, x0 = props.centroid
            session_head_center = (int(y0), int(x0))

            if props.extent < 0.6:
                # need to break up the body from the head
                session_head_parts = session_head.copy()
                session_head_parts[session_head_parts > 0] = 1
                # blob = regionprops(session_head_parts)
                minr, minc, maxr, maxc = props.bbox
                cut_point = session_head_parts.sum(axis=0)[minc+3:maxc-3].argmin() + minc + 3
                # session_head[:, session_head_center[1]] = 0
                session_head[:, cut_point] = 0

        session_head = label(session_head)

        if np.unique(session_head).shape[0] > 2:

            possible_heads = regionprops(session_head)

            all_dist = []
            for props in possible_heads:
                minr, minc, maxr, maxc = props.bbox
                dist = 24
                if minr != 0:
                    y0, x0 = props.centroid
                    dist = distance.euclidean((int(y0), int(x0)), self.cig_center)
                all_dist += [dist]

            min_dist_i = np.argmin(all_dist)
            head_blolb = possible_heads[min_dist_i]
            y0, x0 = head_blolb.centroid
            session_head_center = (int(y0), int(x0))
            h_minr, h_minc, h_maxr, h_maxc = head_blolb.bbox

            # removing anything other than the head
            for props in possible_heads:
                if props != head_blolb:
                    minr, minc, maxr, maxc = props.bbox
                    if minr >= h_minr and minc >= h_minc and maxc <= h_maxc:
                        continue
                    session_head[session_head == props.label] = 0

        session_head[session_head > 0] = 1
        return session_head, session_head_center

    def get_history(self, index):

        hands = self.session_regions.copy()
        hands[hands > 0] = 1
        hands = hands ^ 1
        hands = self.get_human(index) * hands
        hands[0, 0] = 1
        return hands

    def get_history_old(self, index):

        img = self.get_image(index)
        if index < 60:
            return img * 0
        img[img < img.max() - 2] = 0
        for i in range(index - 60, index):
            tmp = self.get_image(i)
            tmp[tmp < tmp.max() - 2] = 0
            img += tmp

        thresholds = threshold_multiotsu(img, 2)
        regions = np.digitize(img, bins=thresholds)

        label_img = label(regions)
        blobs = regionprops(label_img)

        rgb = self.get_colored(index)

        for props in blobs:
            if props.area < 10:
                label_img[label_img == props.label] = 0
                continue

            y0, x0 = props.centroid
            minr, minc, maxr, maxc = props.bbox
            h = maxr - minr
            w = maxc - minc
            x = minr
            y = minc

            x2 = x - 50
            y2 = y - 50

            minr = minr - 3 if minr > 3 else minr
            label_img[minr:maxr + 3, minc - 3:] = 1

        if label_img.sum() == 0:
            label_img[0, 0] = 1

        return label_img


if __name__ == '__main__':
    csv_path = "../../../Data/In-Wild/P1/clean/28_20/data.csv"
    mlx = MLX(csv_path)

    # print(mlx.processs_time())
    #
    i = 1861
    # i = 570

    # history = mlx.get_history(i)
    img = mlx.get_image(i)
    # human = mlx.get_human(i)
    # plt.imshow(human)
    # plt.show()

    plt.imshow(img)
    plt.show()
    # print(img.shape)
