import numpy as np
import cv2
import random
from scipy import ndimage
from skimage.transform import resize




class Random_Temporal_Shuffle(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        Shuffles the temporal channel of the input.

        :param img: np.array): Stack of images

        :return: shuffled input around third axis
        """
        img = np.asarray(img)
        img_dtype = img.dtype
        img = img.astype('float32')

        rand = np.random.randint(0,10)

        if rand < self.p*10:
            img = np.transpose(img, (2,0,1))
            np.random.shuffle(img)

            img = np.transpose(img,(1,2,0))


        img = img.astype(img_dtype)

        return img






class Random_Flip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        Randomly flips the input both vertically and horizantally depending on separate probabilities.
        :param img: np.array): Stack of images

        :return: Flipped H/V
        """
        img = np.asarray(img)
        img_dtype = img.dtype
        img = img.astype('float32')

        rand = np.random.randint(0,10)


        #vertical flip
        if rand < self.p*10:
            img = np.flip(img)


        rand = np.random.randint(0, 10)
        #horizental flip
        if rand < self.p * 10:
            img = np.flipud(img)

        img = img.astype(img_dtype)

        return img


class Random_Crop(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        Randonly CENTERED crops images by size 20x15
        :param img: np.array): Stack of images

        :return: Cropped images
        """
        img = np.asarray(img)
        img_dtype = img.dtype
        img = img.astype('float32')

        rand = np.random.randint(0,10)

        if rand < self.p*10:
            img = self.randomCrop(img, 20, 15)
            img = resize(img, (24, 32, img.shape[2]))
        img = img.astype(img_dtype)

        return img

    def randomCrop(self, img, width, height):

        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        return img


class Blur(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        Blurs the input using gaussian kernel of size 3
        :param img: np.array): Stack of images

        :return: BLured images
        """
        img = np.asarray(img)
        img_dtype = img.dtype
        img = img.astype('float32')

        rand = np.random.randint(0,10)

        if rand < self.p*10:
            #rand_kernel = random.choice([3,5,7])
            rand_kernel = 3
            img = cv2.GaussianBlur(img, (rand_kernel,rand_kernel), cv2.BORDER_DEFAULT)
        img = img.astype(img_dtype)

        return img



class Remove_Cigg_2(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        :param img: np.array): Stack of images

        :return: Normalized image
        """
        img = np.asarray(img)
        img_dtype = img.dtype
        img = img.astype('float32')

        rand = np.random.randint(0,10)
        #print(img.shape)
        if rand < self.p*10:
            for idx in range(0, img.shape[0]):
                im = img[idx].copy()
                mask = np.zeros(im.shape, dtype=np.uint8)
                indices = im>45
                indices = ndimage.binary_dilation(indices, [[True, True, True], [True, True, True], [True, True, True]], iterations=2)
                mask[indices] = 1
                img[idx] = cv2.inpaint(img[idx], mask, 3, flags=cv2.INPAINT_TELEA)



                #print(i,j)
                #img[img>50] = 5
        img = img.astype(img_dtype)

        return img

class Remove_Cigg(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Normalized image
        """
        img = np.asarray(img)
        img_dtype = img.dtype
        img = img.astype('float32')

        rand = np.random.randint(0,10)

        if rand < self.p*10:
            img[img>50] = 26
        img = img.astype(img_dtype)

        return img

class Add_Cigg(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Normalized image
        """
        img = np.asarray(img)
        img_dtype = img.dtype
        img = img.astype('float32')

        rand = np.random.randint(0, 10)

        if rand < self.p * 10:
            rand_x = np.random.randint(10, 22, 5)
            rand_y = np.random.randint(5, 19, 5)
            #temp = np.random.randint(250,260)

            #cig = np.repeat(cig[:, :, np.newaxis], 11, axis=2)
            indices = [0,1,2,3,4,5,6,7,8,9,10]
            indices = random.sample(indices, 5)
            indices = np.array(indices, dtype=np.uint8)
            #cig[:,:,indices] = img[:,:, indices]
            for x in range(rand_x.shape[0]):
                temp = np.random.randint(300, 400)
                cig = self.gaussian_filter((3,3),sigma=1)
                cig = np.ones((3, 3)) * temp * cig
                img[rand_y[x]-1: rand_y[x]+2,rand_x[x]-1:rand_x[x]+2, indices[x]] = cig




        img = img.astype(img_dtype)

        return img

    def gaussian_filter(self, shape=(5, 5), sigma=1):
        x, y = [int(edge / 2) for edge in shape]
        grid = np.array(
            [[((i ** 2 + j ** 2) / (2.0 * sigma ** 2)) for i in range(-x, x + 1)] for j in range(-y, y + 1)])
        g_filter = np.exp(-grid) / (2 * np.pi * sigma ** 2)
        g_filter /= np.sum(g_filter)
        return g_filter

class Jitter_Brightness(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Normalized image
        """
        img = np.asarray(img)
        img_dtype = img.dtype
        img = img.astype('float32')

        rand = np.random.randint(0, 10)

        if rand < self.p * 10:
            rand_dif = np.random.randint(-10,10)
            img += rand_dif

        img = img.astype(img_dtype)

        return img



    def __repr__(self):
        return self.__class__.__name__ + '()'
