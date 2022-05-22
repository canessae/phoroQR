# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from skimage import restoration

class photoDeconv:

    def __init__(self):
        self.flagCircle = False
        self.angle = 0.0
        self.d = 20
        self.snr = 50.0

    def __blur_edge(self, img, d=31):
        h, w  = img.shape[:2]
        img_pad = cv.copyMakeBorder(img, d, d, d, d, cv.BORDER_WRAP)
        img_blur = cv.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
        y, x = np.indices((h, w))
        dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
        w = np.minimum(np.float32(dist)/d, 1.0)
        return img*w + img_blur*(1-w)

    def __motion_kernel(self, angle, d, sz=65):
        kern = np.ones((1, d), np.float32)
        c, s = np.cos(angle), np.sin(angle)
        A = np.float32([[c, -s, 0], [s, c, 0]])
        sz2 = sz // 2
        A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
        kern = cv.warpAffine(kern, A, (sz, sz), flags=cv.INTER_CUBIC)
        return kern

    def __defocus_kernel(self, d, sz=65):
        kern = np.zeros((sz, sz), np.uint8)
        cv.circle(kern, (sz, sz), d, 255, -1, cv.LINE_AA, shift=1)
        kern = np.float32(kern) / 255.0
        return kern

    def blur(self, img, **kwarg):
        if kwarg['d']:
            self.d = kwarg.get("d")

        img_copy = np.float32(img) / 255.0
        filter = self.__defocus_kernel(self.d)
        filter /= filter.sum()
        img_copy = cv.filter2D(img_copy, -1, filter)
        img_copy = img_copy * 255
        #saturation
        img_copy[img_copy>255] = 255
        img_copy[img_copy < 0] = 0
        img_copy = img_copy.astype(np.uint8)
        return img_copy

    def deblur(self, img, **kwargs):
        if 'circle' in kwargs:
            self.flagCircle = True
        if 'angle' in kwargs:
            self.angle = kwargs.get("angle")
        if 'd' in kwargs:
            self.d = kwargs.get("d")
        if 'snr' in kwargs:
            self.snr = kwargs.get("snr")

        img_copy = img.astype(np.float32)
        img_copy /= 255
        img_copy = self.__blur_edge(img_copy)
        IMG = cv.dft(img_copy, flags=cv.DFT_COMPLEX_OUTPUT)

        noise = 10 ** (-0.1 * self.snr)

        if self.flagCircle:
            d = self.d
            psf = self.__defocus_kernel(d)
        else:
            d = self.d
            ang = np.deg2rad(self.angle)
            psf = self.__motion_kernel(ang, d)

        psf /= psf.sum()
        psf_pad = np.zeros_like(img_copy)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = cv.dft(psf_pad, flags=cv.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
        PSF2 = (PSF ** 2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[..., np.newaxis]
        RES = cv.mulSpectrums(IMG, iPSF, 0)
        res = cv.idft(RES, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        res = np.roll(res, -kh // 2, 0)
        res = np.roll(res, -kw // 2, 1)
        res *= 255
        #test lucy richardson
        #res = restoration.richardson_lucy(img_copy, psf, num_iter=2)
        #res *= 255
        # saturation
        res[res > 255] = 255
        res[res < 0] = 0
        res = res.astype(np.uint8)
        return res
