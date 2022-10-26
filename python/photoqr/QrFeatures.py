from configparser import ConfigParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
#import rmsd


class QrFeatures:

    numKeypoints = 5
    threshold = 6.0
    DEBUG = 0

    def __init__(self, numKeypoints, threshold):
        self.numKeypoints = numKeypoints
        self.threshold = threshold

    def __sortDescriptor(self, kp, max_points = numKeypoints):
        kp_sorted = sorted(kp, key=lambda x: x.size, reverse=False)
        items = kp_sorted #get numKeypoints largest sift points

        #debug
        #for ii in kp:
        #    print(str(ii.pt[0])+","+str(ii.pt[1]))

        good = []
        cnt = 0
        lastsize = -10.0
        items_ret = []
        if self.DEBUG == 1:
            print("TOTALE CAMPIONI: "+str(len(kp)))
        while len(good) < max_points*2 and cnt+1 < len(items):
            item = items[-(cnt + 1)]
            if self.DEBUG == 1:
                print("."+str(item.size)+" -> "+str(lastsize))
            if abs(item.size - lastsize) > 2:
                lastsize = item.size
                good.append(int(item.pt[0]))
                good.append(int(item.pt[1]))
                if self.DEBUG == 1:
                    print("size = " + str(item.size))
                    print(str(item.pt[0]) + "," + str(item.pt[1]))
                items_ret.append(item)
            cnt += 1

        '''for item in items:
            good.append(int(item.pt[0]))
            good.append(int(item.pt[1]))
            print(item.size)'''

        if self.DEBUG == 1:
            print("fine " + str(len(good)))
        return items_ret, good

    def showSortedDescriptor(self, img, kp, sort = False, show_plots = False):
        if sort == True:
            items, items_list = self.__sortDescriptor(kp, max_points=20)
        else:
            items = kp

        if show_plots:
            plt.figure(50)
            plt.imshow(img)
        good = []
        for item in items:
            if self.DEBUG == 1:
                print(item.response)
            if show_plots:
                plt.plot(item.pt[0], item.pt[1], "ro")
            good.append(int(item.pt[0]))
            good.append(int(item.pt[1]))
        if show_plots:
            plt.show()

        return good

    def __detect_descriptors(self, imgRGB):
        gray = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        return kp

    def detectDescriptors(self, img):
        return self.__detect_descriptors(img)

    def detectFeatures(self, embedData, photoResized):
        kp = self.__detect_descriptors(photoResized)
        [kp_good, kp_good_list] = self.__sortDescriptor(kp)
        self.showSortedDescriptor(photoResized, kp_good)
        embedData.setFeatures(kp_good_list)
        print(embedData.toJson())

    def __compare_features(self, a, b):
        d = a.shape
        ret = np.zeros((d[0], d[1]))
        goods = [False for i in range(d[0])]
        values = [-1 for i in range(d[0])]
        cnt = 0
        for i in a:
            r = np.outer(np.ones((1, d[0])), i)
            diff = np.sum((r-b)**2, axis=1)
            index = np.argmin(diff)
            err = np.min(diff)
            values[cnt] = err
            ret[cnt, :] = a[index, :]
            if err < self.threshold:
                goods[cnt] = True
            cnt = cnt + 1
            if self.DEBUG == 1:
                print(index)
        return goods, values

    def __compare_features2(self, a, b):
        sizeb = b.shape
        sizea = a.shape
        #ret = np.zeros((d[0], d[1]))
        goods = [False for i in range(sizea[0])]
        values = [-1 for i in range(sizea[0])]
        cnt = 0
        for i in a:
            r = np.outer(np.ones((1, sizeb[0])), i)
            diff = np.sum((r-b)**2, axis=1)
            index = np.argmin(diff)
            err = np.min(diff)
            values[cnt] = err
            #ret[cnt, :] = a[index, :]
            if err < self.threshold:
                goods[cnt] = True
            cnt = cnt + 1
            if self.DEBUG == 1:
                print(index)
        return goods, values

    def __save_plot(self, A, B, filename):

        Ax = A[:, 0]
        Ay = A[:, 1]

        print("AX")
        print(Ax)

        Bx = B[:, 0]
        By = B[:, 1]

        plt.plot(Ax, Ay, "x", markersize=15, linewidth=3)
        plt.plot(Bx, By, "o", markersize=15, linewidth=3)

        plt.grid(True)
        plt.tick_params(labelsize=15)
        plt.show()

    def compareFeatures(self, feaA, feaB):
        A = np.array(feaA, np.float64).reshape((int(len(feaA)/2), 2))
        B = np.array(feaB, np.float64).reshape((int(len(feaB)/2), 2))

        goods, values = self.__compare_features2(A, B)
        print("errore SIFT: " + str(values))
        goods = np.array(goods)
        ret = np.sum(goods==True)

        return ret, A, B


