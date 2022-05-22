import cv2
import matplotlib.patches
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import sys
import imagehash
from PIL import Image
import qrcode
import matplotlib.patches as patches
import Crypto
from Crypto.PublicKey import RSA
from Crypto import Random
from Crypto.Cipher import AES, PKCS1_OAEP
from pyzbar import pyzbar
from skimage import img_as_ubyte
from skimage.transform import swirl

from phothoqr import photoDeconv as deconv

class photoQR:
    __cap = None
    width = -1
    height = 256
    __ratio = 1.33
    __p_height = 165
    __p_width = 120
    p_ratio = __p_height / __p_width
    __hashfunc = staticmethod(imagehash.dhash)
    __threshold = 10
    __tolerance = 60
    #blur parameters
    __d = 20
    __snr = 25
    __border = 5
    __regionToBlur = (50, -50, 50, -50)

    PRIVATE_KEY = "private"
    PUBLIC_KEY = "public"
    NOSWIRL = False
    __swirl_params={
        "radius" : 200,
        "strength" : 5
    }
    BLUR = False

    def __init__(self, privatefile = None, publicfile = None):
        if privatefile is not None:
            self.PRIVATE_KEY = privatefile
        if publicfile is not None:
            self.PUBLIC_KEY = publicfile

    def __decrypt_message(self, msg, cipher):
        default_length = 128
        encrypt_byte = bytes(msg.decode(), 'iso_8859_1')
        length = len(encrypt_byte)
        # print(type(encrypt_byte))
        # print(length)
        # print(encrypt_byte)

        if length <= default_length:
            decrypt_byte = cipher.decrypt(encrypt_byte)
        else:
            offset = 0
            res = []
            while length - offset > 0:
                if length - offset > default_length:
                    res.append(cipher.decrypt(encrypt_byte[offset:offset + default_length]))
                else:
                    res.append(cipher.decrypt(encrypt_byte[offset:]))
                offset += default_length
            decrypt_byte = b''.join(res)
        decrypted = decrypt_byte.decode()
        return decrypted

    def __region_to_blur(self, src, rect):
        dst = src.copy()
        print(rect)
        if len(src.shape) > 2 :
            tmp = dst[rect[0]:rect[1], rect[2]:rect[3], :]
        else:
            tmp = dst[rect[0]:rect[1], rect[2]:rect[3]]
        return tmp

    def __get_region_from_photo(self, img, rect):
        rect = rect.astype(int)
        height = np.max([np.abs(rect[3, 1] - rect[0, 1]), np.abs(rect[2, 1] - rect[1, 1])])
        off_h = int((height * self.__ratio - height) / 2)
        width = np.max([np.abs(rect[1, 0] - rect[0, 0]), np.abs(rect[3, 0] - rect[2, 0])])
        off_w = int((width * self.__ratio - width) / 2)
        dims = img.shape
        lim = np.array([max(rect[0, 1] - 2 * off_h, 0), min(rect[3, 1] + 2 * off_h, dims[0]),
                        max(rect[0, 0] - int(width) - 2 * off_w, 0), min(rect[0, 0]-off_w/2, dims[1])])
        lim = np.array(lim, dtype='uint16')
        region = img[lim[0]:lim[1],
                 lim[2]:lim[3]]
        return region, height, width

    def __extract_image(self, img, rect):
        region, height, width = self.__get_region_from_photo(img, rect)

        plt.figure(22)
        region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

        blur = cv2.medianBlur(region_gray, 5)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
        #thresh = cv2.threshold(sharpen, 100, 255, cv2.THRESH_BINARY_INV)[1]
        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        plt.imshow(close, 'gray')
        ax = plt.gca()
        dims = region.shape
        min_area = dims[0] * dims[1] * 0.3
        max_area = dims[0] * dims[1] * 0.9
        plt.title("found regions number: "+str(len(cnts)))

        prev_area = 0
        ROI = region
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                if prev_area < area:
                    prev_area = area
                ROI = region[y:y + h, x:x + w]
                p = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(p)
        return ROI, height, width

    def __resize(self, img, height=800):
        """ Resize image to given height """
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))

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
        print("compensazione")
        print(srcRect)
        print(dstRect)
        M = cv2.getPerspectiveTransform(srcRect, dstRect)
        dst = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
        dst = cv2.warpPerspective(img.astype(np.uint8), M, dst.shape)
        rect = dstRect
        return dst

    def __generate_keys(self):
        # Generate keys
        random_generator = Random.new().read
        key = RSA.generate(1024, random_generator)  # generate public and private keys

        # Save public key
        # print("public:  " + str(key.publickey().exportKey()))
        fid = open(self.PUBLIC_KEY, "wb")
        fid.write(key.publickey().export_key())
        fid.close()

        # Save private key
        # print("private: " + str(key.export_key()))
        fid = open(self.PRIVATE_KEY, "wb")
        fid.write(key.export_key())
        fid.close()
        return key

    def __read_keys(self):
        fid = open(self.PRIVATE_KEY, "r")
        privatekey = RSA.import_key(fid.read())
        fid.close()
        fid = open(self.PUBLIC_KEY, "r")
        publickey = RSA.import_key(fid.read())
        fid.close()
        return privatekey, publickey

    def __detect_descriptors(self, imgRGB):
        gray = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        '''dst = cv2.drawKeypoints(gray, kp, img)
        plt.imshow(dst)
        plt.show()'''
        return kp, des

    def __compare_images(self, img1, img2):
        kp1, des1 = self.__detect_descriptors(img1)
        kp2, des2 = self.__detect_descriptors(img2)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        #img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
        return matches

    def __show_sorted_descriptors(self, img):
        kp,_ = self.__detect_descriptors(img)
        kp_sorted = sorted(kp, key=lambda x: x.response, reverse=True)
        items = kp_sorted[:20]
        plt.figure(50)
        plt.imshow(img)
        for item in items:
            print(item.pt)
            print(item.response)
            plt.plot(item.pt[0], item.pt[1], "ro")
        plt.show()

    def __create_from_photo(self, photoRGB, privkey = None, check_descriptors = True):

        width = self.__p_width*2
        height = self.__p_height*2
        photoResized = cv2.resize(photoRGB, (width, height))
        print(photoResized.shape)
        if self.NOSWIRL:
            photoResized = img_as_ubyte(photoResized)
        else:
            photoResized = img_as_ubyte(swirl(photoResized,
                                              rotation=0,
                                              strength = self.__swirl_params['strength'],
                                              radius = self.__swirl_params['radius']))
        # evaluate hash from resized image
        hash = self.__hashfunc(Image.fromarray(photoResized[self.__border:-self.__border, self.__border:-self.__border, :]))
        print("hash to be saved in QR = " + str(str(hash).encode()))
        #blur if necessary and border
        if self.BLUR:
            rect = self.__regionToBlur
            region = self.__region_to_blur(photoResized, rect)
            blurred = self.blur(region)
            photoResized[rect[0]:rect[1], rect[2]:rect[3], :] = blurred
            photoResized = photoResized[self.__border:-self.__border, self.__border:-self.__border, :]
        else:
            photoResized = photoResized[self.__border:-self.__border,self.__border:-self.__border,:]
        rsa = PKCS1_OAEP.new(privkey)
        if privkey is None:
            hash_enc = hash
        else:
            hash_enc = rsa.encrypt(str(hash).encode())
        # print(rsa.encrypt(str(hash).encode(), privkey).hex())
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
        # Create descriptors with SIFT
        if check_descriptors:
            kp, des = self.__detect_descriptors(photoResized)

        #self.__show_sorted_descriptors(photoResized)

        #added black borders here, see limits
        dst[self.__border:height-self.__border, self.__border:width-self.__border, :] = photoResized
        dst[0:height, width:width + height, :] = code_resized
        return dst, hash

    def blur(self, img):
        b = deconv.photoDeconv()
        res = np.zeros(img.shape, np.uint8)
        if len(img.shape) == 3:
            for i in range(0, img.ndim):
                res[:,:,i] = b.blur(img[:,:,i], d = self.__d)
        else:
            res[:, :] = b.blur(img[:, :], d=self.__d)
        return res

    def deblur(self, img):
        b = deconv.photoDeconv()
        res = np.zeros(img.shape, np.uint8)
        res[self.__border:-self.__border,self.__border:-self.__border] = img[self.__border:-self.__border,self.__border:-self.__border,:].copy()
        if len(img.shape) == 3:
            for i in range(0, img.ndim):
                rect = self.__regionToBlur
                deblurred = b.deblur(img[rect[0]:rect[1], rect[2]:rect[3], i], d = self.__d, circle=True, snr = self.__snr)
                res[rect[0]:rect[1], rect[2]:rect[3], i] = deblurred
        else:
            res[self.__border:-self.__border, self.__border:-self.__border] = b.deblur(img[self.__border:-self.__border, self.__border:-self.__border], d=self.__d, circle=True, snr=self.__snr)
        return res

    # Method to iteratively detect the QR
    def detect(self, dev = 0, photo = None, privatekey = None, check_descriptors = False, wait_correct = False):
        data = None
        if photo is None:
            self.__cap = cv2.VideoCapture(dev)
            if self.__cap is None or not self.__cap.isOpened():
                print("Camera not found!")
                sys.exit(-1)

        # Prepare for decrypting
        if (privatekey is not None):
            key = privatekey
            cipher = PKCS1_OAEP.new(key)

        # initialize the cv2 QRCode detector
        fig = plt.figure(1)
        if photo is not None:
            img = photo
        else:
            _, img = self.__cap.read()
        img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ax = plt.gca()
        im = ax.imshow(img_conv)

        while True:
            if photo is not None:
                img = photo
            else:
                _, img = self.__cap.read()
            img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            im.set_data(img_conv)
            # plt.draw()
            # check if button is pressed
            if wait_correct == True and photo is None:
                ret = plt.waitforbuttonpress(0.00001)
                if ret != True:
                    continue
            else:
                plt.pause(0.001)
            # detect and decode
            # data, bbox, _ = detector.detectAndDecode(img)
            ddd = pyzbar.decode(img)
            if len(ddd) == 0:
                continue
            data = ddd[0].data
            bbox = ddd[0].polygon
            print(ddd[0].polygon)
            print(len(data.decode()))

            # check if there is a QRCode in the image
            if data:
                if (privatekey is not None):
                    data = self.__decrypt_message(data, cipher)
                else:
                    data = data.decode()
                break

        # Release camera
        if self.__cap is not None:
            self.__cap.release()

        if not data:
            return False

        plt.subplot(1, 2, 1)
        plt.imshow(img_conv);
        plt.axis(False)
        hash_from_qr = imagehash.hex_to_hash(str(data))
        plt.title("Hash from QR: " + str(hash_from_qr))
        plt.plot(bbox[0].x, bbox[0].y, "r+")#tl
        plt.plot(bbox[1].x, bbox[1].y, "b+")#bl
        plt.plot(bbox[2].x, bbox[2].y, "g+")#br
        plt.plot(bbox[3].x, bbox[3].y, "y+")#tr

        rect = np.array([[bbox[0].x, bbox[0].y],
                         [bbox[3].x, bbox[3].y],
                         [bbox[2].x, bbox[2].y],
                         [bbox[1].x, bbox[1].y]], 'float32')
        #
        plt.figure(2)
        compensated = cv2.cvtColor(self.__compensate_perspective(img, rect), cv2.COLOR_BGR2RGB)
        plt.title("Compensated")
        plt.imshow(compensated)
        #

        plt.figure(1)
        plt.subplot(1, 2, 2)
        region, _, _ = self.__extract_image(compensated, rect)
        w = self.__p_width*2
        h = self.__p_height*2
        region = cv2.resize(region, (w, h))
        plt.figure(1)
        # if blur is needed
        if self.BLUR:
            region = self.deblur(region)
        # if swirl enabled
        if self.NOSWIRL:
            plt.imshow(region)
        else:
            region_orig = region.copy()
            region = swirl(region,
                           rotation=0,
                           strength = -self.__swirl_params['strength'],
                           radius = self.__swirl_params['radius'])
            region = img_as_ubyte(region)
            plt.imshow(region)
        #test
        #self.__show_sorted_descriptors(region)
        hash_from_image = self.__hashfunc(Image.fromarray(region_orig[self.__border:-self.__border,self.__border:-self.__border,:]))
        print("Similarity = " + str(hash_from_image - hash_from_qr))
        plt.title(str(hash_from_image))
        plt.axis(False)
        dims = region.shape
        if hash_from_image - hash_from_qr <= self.__threshold:
            result = cv2.imread("correct.png")
        else:
            result = cv2.imread("wrong.png")

        result = cv2.resize(result, (dims[1], dims[0]))
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        #plt.imshow(result, alpha=0.3, interpolation='bilinear')
        plt.show()
        return True

    def create(self, photo = None, privkey = None):
        photoRGB = photo
        dst, hash = self.__create_from_photo(photoRGB, privkey, True)
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
        dx = self.__p_width * 2
        dy = self.__p_height * 2
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

