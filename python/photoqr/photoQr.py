import json
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import imagehash
import qrcode
import matplotlib.patches as patches
from Crypto.Cipher import AES, PKCS1_OAEP
from pyzbar import pyzbar
from skimage import img_as_ubyte

from photoqr.EmbeddedData import EmbeddedData
from photoqr.QrEncrypt import QrEncrypt
from photoqr.QrFeatures import QrFeatures
from photoqr.QrWarp import QrWarp
from configparser import ConfigParser

# defining a helper class for implementing multi-threading
class WebcamStream:
    # initialization method
    def __init__(self, stream_id = 0):
        self.stream_id = stream_id  # default is 0 for main camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        #self.vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        #self.vcap.set(cv2.CAP_PROP_FPS, 10)
        #self.vcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

    # method to return latest read frame
    def read(self):
        ret, frame = self.vcap.read()
        if ret:
            self.frame = frame
        return self.frame.copy()

    def start(self):
        pass

    # method to stop reading frames
    def stop(self):
        self.vcap.release()

    def getStreamId(self):
        return self.stream_id

class photoQR:

    __general_config = {
        "width" : -1,
        "height" : 256,
        "ratio" : 1.33,
        "p_height" : 165,
        "p_width" : 120,
        "threshold" : 10,
        "threshold_sift" : 3,
        "tolerance" : 60
    }

    __hashfunc = staticmethod(imagehash.dhash)

    # encryption params
    ENCRYPT = False
    __enc_params = {
        "PRIVATE_KEY" : "private",
        "PUBLIC_KEY" : "public"
    }

    # swirl effect
    SWIRL = False
    __swirl_params = {
        "radius" : 200,
        "strength" : 5
    }

    # blurring effect
    BLUR = False
    __blur_param = {
        "flagCircle": True,
        "angle": 0.0,
        "diameter": 20,
        "snr": 25.0,
        "regionToBlur" :  (50, -50, 50, -50),
        "border" : 7
    }

    __colors = {
        "HEADER" : '\033[95m',
        "OKBLUE" : '\033[94m',
        "OKCYAN" : '\033[96m',
        "OKGREEN" : '\033[92m',
        "WARNING" : '\033[93m',
        "FAIL" : '\033[91m',
        "ENDC" : '\033[0m',
        "BOLD" : '\033[1m',
        "UNDERLINE" : '\033[4m'
    }

    __features = {
        'numKeypoints' : 5,
        'threshold' : 6.0
    }

    __DEBUG = 0

    def __importConfigFromFile(self, filename : str = "config.ini"):
        # Read config.ini file
        config_object = ConfigParser()
        ret = config_object.read(filename)
        if len(ret) > 0:
            general = config_object['GENERAL']
            #self.BLUR = general.getboolean('enable_blur', False)
            self.BLUR = False
            self.SWIRL = general.getboolean('enable_swirl', False)
            self.ENCRYPT = general.getboolean('enable_crypt', False)
            self.__general_config["threshold"] = general.getfloat('threshold', 10.0)
            self.__general_config["threshold_sift"] = general.getint('threshold_sift', 3)
            encrypt = config_object['ENCRYPT']
            self.__enc_params['PRIVATE_KEY'] = encrypt['private_key']
            self.__enc_params['PUBLIC_KEY'] = encrypt['public_key']
            swirl = config_object['SWIRL']
            self.__swirl_params['radius'] = swirl.getint('radius', 200)
            self.__swirl_params['strength'] = swirl.getint('strength', 5)
            features = config_object['FEATURES']
            self.__features['numKeypoints'] = features.getint('num_keypoints', 5)
            self.__features['threshold'] = features.getfloat('threshold', 6.0)

    def __init__(self):
        # Read configuration from file
        self.__importConfigFromFile()

        # warp
        self.warp = QrWarp(self.__swirl_params,
                           self.__blur_param)
        # encryot
        self.encrypt = QrEncrypt(self.__enc_params)

        # general
        self.width = self.__general_config["width"]
        self.height = self.__general_config["height"]
        self.p_ratio = self.__general_config["p_height"] / self.__general_config["p_width"]

    def __get_region_from_photo(self, img, rect):
        rect = rect.astype(int)
        height = np.max([np.abs(rect[3, 1] - rect[0, 1]), np.abs(rect[2, 1] - rect[1, 1])])
        off_h = int((height * self.__general_config["ratio"] - height) / 2)
        width = np.max([np.abs(rect[1, 0] - rect[0, 0]), np.abs(rect[3, 0] - rect[2, 0])])
        off_w = int((width * self.__general_config["ratio"] - width) / 2)
        dims = img.shape
        lim = np.array([max(rect[0, 1] - 2 * off_h, 0), min(rect[3, 1] + 2 * off_h, dims[0]),
                        max(rect[0, 0] - int(width) - 2 * off_w, 0), min(rect[0, 0]-off_w/2, dims[1])])
        lim = np.array(lim, dtype='uint16')
        region = img[lim[0]:lim[1],
                 lim[2]:lim[3]]
        return region, height, width

    def __extract_image(self, img, rect):
        region, height, width = self.__get_region_from_photo(img, rect)

        if self.__DEBUG == 1:
            plt.figure(22)
        region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

        blur = cv2.medianBlur(region_gray, 5)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
        #thresh = cv2.threshold(sharpen, 100, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

        cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        if self.__DEBUG == 1:
            plt.imshow(close, 'gray')
            ax = plt.gca()
        dims = region.shape
        min_area = dims[0] * dims[1] * 0.3
        max_area = dims[0] * dims[1] * 0.98
        if self.__DEBUG == 1:
            plt.title("found regions number: "+str(len(cnts)))

        prev_area = 0
        ROI = region
        for c in cnts:
            area = cv2.contourArea(c)
            #print(". "+ str(area) +","+str(max_area))
            if area > min_area and area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                if prev_area < area:
                    prev_area = area
                ROI = region[y:y + h, x:x + w]
                p = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                if self.__DEBUG == 1:
                    ax.add_patch(p)
        return ROI, height, width

    def __resize(self, img, height=800):
        """ Resize image to given height """
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))

    def __find_image_area(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        s = img_gray.shape
        ret, thresh = cv2.threshold(img_gray, 150, 255, 0)
        im = thresh
        contours, hierarchy  = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        goodContours = []
        for cnt in contours:
            #print(".")
            # Simplify contour
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
            # Page has 4 corners and it is convex
            # Page area must be bigger than maxAreaFound
            if (len(approx) == 4 and
                    cv2.isContourConvex(approx) and
                    cv2.contourArea(approx) > 0.2*s[0]*s[1] and
                    cv2.contourArea(approx) < 0.8*s[0]*s[1]):
                areas.append(cv2.contourArea(approx))
                goodContours.append(approx)

        if not areas:
            print("Nothing found")
            return None
        bestArea = max(areas)
        indexBestArea = areas.index(bestArea)
        bestContour = np.array(goodContours[indexBestArea], np.int32)

        print("faccio vedere area")
        print(len(areas))
        print(areas)
        print(indexBestArea)
        #img = cv2.drawContours(image, goodContours, -1, (0,255,75), 2)
        #plt.imshow(thresh, cmap='gray')
        #plt.imshow(img)
        #plt.show()
        boundingRect = cv2.boundingRect(bestContour)
        print(boundingRect)
        return bestContour, boundingRect

    def __extract_image2(self, src, rect):
        region, height, width = self.__get_region_from_photo(src, rect)
        # Resize and convert to grayscale
        img = cv2.cvtColor(self.__resize(region), cv2.COLOR_BGR2GRAY)
        # Bilateral filter preserv edges
        img = cv2.bilateralFilter(img, 9, 75, 75)
        # Create black and white image based on adaptive threshold
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
        # Median filter clears small details
        img = cv2.medianBlur(img, 11)
        # Add black border in case that page is touching an image border
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        edges = cv2.Canny(img, 200, 250)
        plt.figure(55)
        plt.imshow(region)
        plt.figure(55)
        plt.imshow(edges)
        plt.show()
        # Getting contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Finding contour of biggest rectangle
        # Otherwise return corners of original image
        # Don't forget on our 5px border!
        height = edges.shape[0]
        width = edges.shape[1]
        MAX_COUNTOUR_AREA = (width - 10) * (height - 10)
        # Page fill at least half of image, then saving max area found
        maxAreaFound = MAX_COUNTOUR_AREA * 0.5
        # Saving page contour
        pageContour = np.array([[5, 5], [5, height - 5], [width - 5, height - 5], [width - 5, 5]])
        # Go through all contours
        for cnt in contours:
            # Simplify contour
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
            # Page has 4 corners and it is convex
            # Page area must be bigger than maxAreaFound
            if (len(approx) == 4 and
                    cv2.isContourConvex(approx) and
                    maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
                maxAreaFound = cv2.contourArea(approx)
                pageContour = approx
        print("resuslts:")
        print(pageContour)

    def __compensate_perspective(self, img, rect):
        height = np.abs(rect[3, 1] - rect[0, 1])
        width = np.abs(rect[1, 0] - rect[0, 0])
        dim = int(max(height, width))
        srcRect = rect
        dstRect = np.float32([[srcRect[0, 0], srcRect[0, 1]],
                              [srcRect[0, 0] + dim, srcRect[0, 1]],
                              [srcRect[0, 0] + dim, srcRect[0, 1] + dim],
                              [srcRect[0, 0], srcRect[0, 1] + dim]])
        '''print("compensazione")
        print(srcRect)
        print(dstRect)'''
        M = cv2.getPerspectiveTransform(srcRect, dstRect)
        dst = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
        dst = cv2.warpPerspective(img.astype(np.uint8), M, dst.shape)
        rect = dstRect
        return dst

    def __create_from_photo(self, photoRGB, privkey = None, check_descriptors = True):
        width = self.__general_config["p_width"]*2
        height = self.__general_config["p_height"]*2
        photoResized = cv2.resize(photoRGB, (width, height))
        print(photoResized.shape)
        if not self.SWIRL:
            photoResized = img_as_ubyte(photoResized)
        else:
            photoResized = self.warp.applySwirl(photoResized)

        # evaluate hash from resized image
        hash = self.__hashfunc(Image.fromarray(photoResized[self.__blur_param["border"]:-self.__blur_param["border"], self.__blur_param["border"]:-self.__blur_param["border"], :]))
        print("hash to be saved in QR = " + str(str(hash).encode()))
        embedData = EmbeddedData(hash = str(hash))
        #blur if necessary and border
        if self.BLUR:
            blurred = self.warp.applyBlur(photoResized)
            rect = self.warp.getRegionToBlur()
            #region = self.__region_to_blur(photoResized, rect)
            #blurred = self.blur(region)
            photoResized[rect[0]:rect[1], rect[2]:rect[3], :] = blurred
            photoResized = photoResized[self.__blur_param["border"]:-self.__blur_param["border"], self.__blur_param["border"]:-self.__blur_param["border"], :]
        else:
            photoResized = photoResized[self.__blur_param["border"]:-self.__blur_param["border"], self.__blur_param["border"]:-self.__blur_param["border"],:]
        rsa = PKCS1_OAEP.new(privkey)

        ## SIFT if necessary
        # Create descriptors with SIFT
        if check_descriptors:
            features = QrFeatures(self.__features['numKeypoints'], self.__features['threshold'])
            features.detectFeatures(embedData, photoResized)

        # Check encryption if necessary
        if privkey is None:
            hash_enc = embedData.toJson()
        else:
            hash_enc = rsa.encrypt(str(hash).encode())
        # print(rsa.encrypt(str(hash).encode(), privkey).hex())

        # Create QR image
        code = qrcode.make(hash_enc)
        tmp = np.array(code, dtype='uint8') * 255
        ttt = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
        # rileva proporzione
        dove = np.argwhere(tmp == 0)
        qrsize = tmp.shape
        delta = dove[-1, 0] - dove[0, 0]
        ratio = qrsize[0] / delta
        # print(str(ratio))
        # ttt = ttt[dove[0,0]:dove[-1,0]+1,dove[0,1]:dove[-1,1]+1]
        code_resized = cv2.resize(ttt, dsize=(height, height), interpolation=cv2.INTER_NEAREST_EXACT)
        dst = np.zeros((height, width + height, 3), 'uint8')

        #added black borders here, see limits
        dst[self.__blur_param["border"]:height-self.__blur_param["border"], self.__blur_param["border"]:width-self.__blur_param["border"], :] = photoResized
        dst[0:height, width:width + height, :] = code_resized
        return dst, hash

    def __areaQr(self, bbox):
        tl = np.array([bbox[0].x, bbox[0].y], np.float32)
        bl = np.array([bbox[1].x, bbox[1].y], np.float32)
        br = np.array([bbox[2].x, bbox[2].y], np.float32)
        tr = np.array([bbox[3].x, bbox[3].y], np.float32)
        v1m = ((tl-bl)+(tr-br))/2
        v2m = ((br-bl)+(tr-tl))/2
        area = np.linalg.det([v1m, v2m])
        '''print("area")
        print(tl)
        print(bl)
        print(br)
        print(tr)
        print(v1m)
        print(v2m)
        print(area)'''
        return area

    def __check_qr_is_present(self, img):
        area_img = img.shape[0] * img.shape[1]
        ddd = pyzbar.decode(img)
        if len(ddd) == 0:
            return None
        data = ddd[0].data
        bbox = ddd[0].polygon
        if self.__areaQr(bbox) / area_img < 0.10:
            return None

        return bbox, data

    def __from_bbox2rect(self, bbox):
        rect = np.array([[bbox[0].x, bbox[0].y],
                         [bbox[3].x, bbox[3].y],
                         [bbox[2].x, bbox[2].y],
                         [bbox[1].x, bbox[1].y]], 'float32')
        return rect

    # Method to iteratively detect the QR
    def detect(self, webcam : WebcamStream = None, dev = 0, photo = None, privatekey = None, check_descriptors = False, wait_correct = False, forceExitWithButton = False):
        data = None

        # initialize the cv2 QRCode detector
        if not plt.fignum_exists(1):
            fig = plt.figure(1)
            plt.axis(False)
            if photo is not None:
                img = photo
            else:
                img = webcam.read()
            img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ax = plt.gca()
            self.im = ax.imshow(img_conv)

        #lasttime = datetime.now()
        while True:
            if photo is not None:
                img = photo
            else:
                img = webcam.read()
            img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.im.set_data(img_conv)
            #currenttime = datetime.now()
            #print("t0: " + str(currenttime-lasttime))
            #lasttime = currenttime

            # plt.draw()
            # check if button is pressed
            if wait_correct and photo is None:
                ret = plt.waitforbuttonpress(0.00001)
                if ret != True:
                    continue
            elif forceExitWithButton:
                ret = plt.waitforbuttonpress(0.00001)
                if ret:
                    return False
            else:
                plt.pause(0.001)

            # detect and decode
            res = self.__check_qr_is_present(img)
            if res is not None:
                bbox = res[0]
                data = res[1]
                # debug
                print(len(data.decode()))

            # check if there is a QRCode in the image
            if data:
                if (self.ENCRYPT is True):
                    data = self.encrypt.decryptMessage(data)
                else:
                    data = data.decode()
                break

        if not data:
            return False
        if self.__DEBUG == 1:
            plt.subplot(1, 2, 1)
            plt.imshow(img_conv)
            plt.axis(False)

        #temporary
        print(data)
        raw = json.loads(data)
        data = raw["hash_code"]
        featFromQr = raw["features"]
        #
        hash_from_qr = imagehash.hex_to_hash(str(data))
        if self.__DEBUG == 1:
            plt.title("Hash from QR: " + str(hash_from_qr))
            plt.plot(bbox[0].x, bbox[0].y, "r+")#tl
            plt.plot(bbox[1].x, bbox[1].y, "b+")#bl
            plt.plot(bbox[2].x, bbox[2].y, "g+")#br
            plt.plot(bbox[3].x, bbox[3].y, "y+")#tr

        rect = self.__from_bbox2rect(bbox)
        #
        compensated = cv2.cvtColor(self.__compensate_perspective(img, rect), cv2.COLOR_BGR2RGB)
        if self.__DEBUG == 1:
            plt.figure(2)
            plt.title("Compensated")
            plt.imshow(compensated)
        #

        # improve image detection
        if photo is None:
        #    plt.imsave("temporary.png", img_conv)
            ret = self.__find_image_area(compensated)
            if ret is not None:
                best = ret[0]
                r = ret[1]
                img = compensated[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                if self.__DEBUG == 1:
                    plt.title("Compensated and cut")
                    plt.imshow(img)
                    plt.show()
                img = cv2.resize(img, (590, 350))
                ddd = pyzbar.decode(img)
                if len(ddd) == 0:
                    return True #go ahead and retry
                print("data size: " + str(len(ddd)))
                bbox = ddd[0].polygon
                rect = self.__from_bbox2rect(bbox)
                compensated = img

        if self.__DEBUG == 1:
            plt.figure(100)
            plt.imshow(compensated)
            plt.figure(1)
            plt.subplot(1, 2, 2)

        region, _, _ = self.__extract_image(compensated, rect)
        w = self.__general_config["p_width"]*2
        h = self.__general_config["p_height"]*2
        region = cv2.resize(region, (w, h))

        # if blur is needed
        if self.BLUR:
            region = self.warp.applyBlur(region, reverse=True)
        # if swirl enabled
        if not self.SWIRL:
            region_orig = region.copy()
        else:
            region_orig = region.copy()
            region = self.warp.applySwirl(region, reverse=True)
            region = img_as_ubyte(region)

        if self.__DEBUG == 1:
            plt.figure(1)
            plt.imshow(region)

        #test
        features = QrFeatures(self.__features['numKeypoints'], self.__features['threshold'])
        kpn = features.detectDescriptors(region_orig[self.__blur_param["border"]:-self.__blur_param["border"],self.__blur_param["border"]:-self.__blur_param["border"],:])
        kp_good = features.showSortedDescriptor(region_orig[self.__blur_param["border"]:-self.__blur_param["border"],self.__blur_param["border"]:-self.__blur_param["border"],:], kpn, True)
        print("SIFT features from QR  : " + str(featFromQr))
        print("SIFT from current image: " + str(kp_good))
        if len(kp_good) == 0:
            return True
        res, A, B = features.compareFeatures(featFromQr, kp_good)
        oneTwo = 0
        if res >= self.__general_config["threshold_sift"]:
            print(self.__colors["OKGREEN"] + "SIFT matching:   " + str(res) + "/" + str(int(len(featFromQr)/2)) + self.__colors["ENDC"])
            oneTwo += 1
        else:
            print(self.__colors["WARNING"] + "SIFT matching:   " + str(res) + "/" + str(int(len(featFromQr)/2)) + self.__colors["ENDC"])

        hash_from_image = self.__hashfunc(Image.fromarray(region_orig[self.__blur_param["border"]:-self.__blur_param["border"], self.__blur_param["border"]:-self.__blur_param["border"],:]))
        simil = hash_from_image - hash_from_qr
        if simil <= self.__general_config["threshold"]:
            print(self.__colors["OKGREEN"] + "HASH Similarity: " + str(simil) + self.__colors["ENDC"])
            oneTwo += 1
        else:
            print(self.__colors["WARNING"] + "HASH Similarity: " + str(simil) + self.__colors["ENDC"])

        #Show when found correct points and hash
        if oneTwo == 2:
            f = plt.figure(50)
            plt.imshow(region)
            plt.title(str(hash_from_image)+"/"+str(hash_from_qr))
            for ii in A:
                plt.plot(A[:,0], A[:,1],'xr')
            for ii in B:
                plt.plot(B[:, 0], B[:, 1], 'ob', mfc='none')
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close(f)

        if self.__DEBUG == 1:
            plt.title(str(hash_from_image))
            plt.axis(False)
        dims = region.shape
        if hash_from_image - hash_from_qr <= self.__general_config["threshold"]:
            result = cv2.imread("correct.png")
        else:
            result = cv2.imread("wrong.png")

        result = cv2.resize(result, (dims[1], dims[0]))
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        #plt.imshow(result, alpha=0.3, interpolation='bilinear')
        if self.__DEBUG == 1:
            plt.show()

        return True

    def create(self, photo = None, privkey = None, show_plots = False):
        photoRGB = photo
        dst, hash = self.__create_from_photo(photoRGB, privkey, True)
        if show_plots:
            plt.figure(1)
            plt.imshow(dst)
            plt.axis(False)
            #plt.title(str(hash))
            plt.show()
        nimg = np.ones((dst.shape[0] + 20, dst.shape[1] + 20, 3), dtype=np.uint8) * 255
        nimg[10:dst.shape[0] + 10, 10:dst.shape[1] + 10, :] = dst
        cv2.imwrite("Reference.png", cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR))

    def acquire_video_until_button(self, dev=0):
        cap = cv2.VideoCapture(dev)

        if cap is None or not cap.isOpened():
            print("Camera not found!")
            return None

        fig = plt.figure(1)
        _, img = cap.read()
        dims = img.shape
        dx = self.__general_config["p_width"] * 2
        dy = self.__general_config["p_height"] * 2
        x = int((dims[1] - dx) / 2)
        y = int((dims[0] - dy) / 2)
        ax = plt.gca()
        img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        im = ax.imshow(img_conv)
        plt.ion()
        while True:
            _, img = cap.read()
            img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            im.set_data(img_conv)
            p = plt.Rectangle((x, y), dx, dy, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(p)
            ret = plt.waitforbuttonpress(0.00001)
            if ret == True:
                break
        plt.ioff()
        cap.release()
        return img_conv[y:y + dy, x:x + dx, :]

if __name__ == "__main__":
    '''tmp = photoQR()
    q = Queue()
    cmd = Queue()
    producer = Process(target=tmp.produce_frames, args=(q, cmd, 0,))
    producer.start()
    tmp.consume_frames(q)'''
    pass