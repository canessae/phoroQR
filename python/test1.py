import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import imagehash
from PIL import Image

boxsize=3

def test():
    img = cv2.imread("/home/ltenze/progetti/qr/AR_warp_zip/test2/M-001-01.bmp")
    #img = cv2.imread("/home/ltenze/progetti/qr/AR_warp_zip/test2/M-050-26.bmp")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(1)
    plt.imshow(gray, 'gray')
    hash1 = imagehash.dhash(Image.fromarray(gray))
    plt.figure(2)
    _,bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU);
    plt.imshow(bw, 'gray')
    #
    hash2 = imagehash.dhash(Image.fromarray(bw))
    print(str(hash1)+"\n"+str(hash2))
    dims = bw.shape
    #print(dims)
    nuova = np.zeros((math.floor(dims[0]/boxsize), math.floor(dims[1]/boxsize)))
    #print(nuova.shape)

    y = 0
    for i in range(0, dims[0], boxsize):
        x = 0
        for j in range (0, dims[1], boxsize):
            nuova[y,x] = bw[i,j]
            #print(str(x)+"-"+str(y))
            x+=1
        y+=1

    plt.figure(3)
    plt.imshow(nuova, 'gray')
    hash3 = imagehash.dhash(Image.fromarray(nuova))
    print(str(hash3))
    print(nuova.shape)
    print(str(hash1-hash2))
    print(str(hash2 - hash3))
    print(str(hash1 - hash3))
    plt.show()

if __name__ == '__main__':
    test()