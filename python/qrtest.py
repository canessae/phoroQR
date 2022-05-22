import cv2
import qrcode
import matplotlib.pyplot as plt
import sys

def create():
    img = qrcode.make(sys.argv[2])
    img.save("saved.png")

def read():
    img = cv2.imread("saved.png")
    det = cv2.QRCodeDetector()
    val,pts,st_code = det.detectAndDecode(img)
    print(val)

if __name__ == "__main__":
    if sys.argv[1] == 'create':
        create()
    elif sys.argv[1] == 'read':
        read()