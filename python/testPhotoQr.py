import getopt
import sys
from os.path import exists

import cv2
import matplotlib.pyplot as plt
import numpy as np
import photoqr
from photoqr.photoQr import WebcamStream

sys.path.append(r'/home/ltenze/progetti/qr/python/photoqr')

white_border = 20

if __name__ == "__main__":
    test = photoqr.photoQR()
    '''try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print
        'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)'''
    if len(sys.argv)>2 and sys.argv[1] == "create" and exists(sys.argv[2]):
        img = cv2.imread(sys.argv[2])
        img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test.create(img_conv)
    elif len(sys.argv)>2 and sys.argv[1] == "create": #webcam device
        img = test.acquire_video_until_button(dev=int(sys.argv[2]))
        test.create(img)
    elif sys.argv[1] == "create":
        img = test.acquire_video_until_button()
        test.create(img)
    elif len(sys.argv)>2 and sys.argv[1] == "detect" and exists(sys.argv[2]):
        img = cv2.imread(sys.argv[2])
        nimg = np.ones((img.shape[0]+white_border*2, img.shape[1]+white_border*2, 3), dtype=np.uint8)*255
        nimg[white_border:img.shape[0] + white_border, white_border:img.shape[1] + white_border, :] = img
        test.detect(photo=nimg)
    elif sys.argv[1] == "test":
        orig = cv2.imread("Reference.png", cv2.IMREAD_COLOR)
        #orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        plt.figure(1)
        plt.title("original")
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        dst = test.blur(orig)
        plt.figure(2)
        plt.title("blurred")
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        dst = orig
        plt.figure(3)
        plt.title("deblurred")
        res = test.deblur(dst)
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.show()

    elif len(sys.argv)>2 and sys.argv[1] == "detect":
        print(sys.path)
        webcam = WebcamStream(stream_id=int(sys.argv[2]))
        webcam.start()
        while True:
            if not test.detect(dev=int(sys.argv[2]), forceExitWithButton=True, webcam = webcam):
                break
        webcam.stop()
    else:
        webcam = WebcamStream()
        webcam.start()
        while True:
            if not test.detect(forceExitWithButton=True, webcam = webcam):
                break
        webcam.stop()


