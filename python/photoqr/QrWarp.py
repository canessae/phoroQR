import numpy as np
from skimage import img_as_ubyte
from skimage.transform import swirl

import sys
sys.path.append(r'photoqr')
import photoDeconv as deconv

class QrWarp:

    __swirl_params = {
        "radius": 200,
        "strength": 5
    }

    __blur_param = {
        "flagCircle": True,
        "angle": 0.0,
        "diameter": 20,
        "snr": 25.0,
        "regionToBlur" :  (50, -50, 50, -50),
        "border" : 5
    }

    def __init__(self, swirl_params, blur_param):
        self.__swirl_params = swirl_params
        self.__blur_param = blur_param

    def __blur(self, img):
        b = deconv.photoDeconv()
        res = np.zeros(img.shape, np.uint8)
        if len(img.shape) == 3:
            for i in range(0, img.ndim):
                res[:, :, i] = b.blur(img[:, :, i], d = self.__blur_param["diameter"])
        else:
            res[:, :] = b.blur(img[:, :], d=self.__blur_param["diameter"])
        return res

    def __region_to_blur(self, src, rect):
        dst = src.copy()
        if len(src.shape) > 2 :
            tmp = dst[rect[0]:rect[1], rect[2]:rect[3], :]
        else:
            tmp = dst[rect[0]:rect[1], rect[2]:rect[3]]
        return tmp

    def __deblur(self, img):
        b = deconv.photoDeconv()
        res = np.zeros(img.shape, np.uint8)
        res[self.__blur_param["border"]:-self.__blur_param["border"], self.__blur_param["border"]:-self.__blur_param["border"]] = \
            img[self.__blur_param["border"]:-self.__blur_param["border"], self.__blur_param["border"]:-self.__blur_param["border"],:].copy()
        if len(img.shape) == 3:
            for i in range(0, img.ndim):
                rect = self.__blur_param["regionToBlur"]
                deblurred = b.deblur(img[rect[0]:rect[1], rect[2]:rect[3], i],
                                     d = self.__blur_param["diameter"],
                                     circle = self.__blur_param["flagCircle"],
                                     snr = self.__blur_param["snr"])
                res[rect[0]:rect[1], rect[2]:rect[3], i] = deblurred
        else:
            res[self.__blur_param["border"]:-self.__blur_param["border"], self.__blur_param["border"]:-self.__blur_param["border"]] = \
                b.deblur(img[self.__blur_param["border"]:-self.__blur_param["border"], self.__blur_param["border"]:-self.__blur_param["border"]],
                         d = self.__blur_param["diameter"],
                         circle = self.__blur_param["flagCircle"],
                         snr = self.__blur_param["snr"])
        return res

    def applySwirl(self, imgCv2, reverse = False):
        if reverse:
            ret = img_as_ubyte(swirl(imgCv2,
                                     rotation = 0,
                                     strength = - self.__swirl_params['strength'],
                                     radius = self.__swirl_params['radius']))
        else:
            ret = img_as_ubyte(swirl(imgCv2,
                                     rotation = 0,
                                     strength = self.__swirl_params['strength'],
                                     radius = self.__swirl_params['radius']))
        return ret

    def applyBlur(self, imgCv2, reverse = False):

        if reverse:
            ret = self.__deblur(imgCv2)
        else:
            imgCv2 = self.__region_to_blur(imgCv2, self.__blur_param["regionToBlur"])
            ret = self.__blur(imgCv2)

        return ret

    def getRegionToBlur(self):
        return self.__blur_param["regionToBlur"]
