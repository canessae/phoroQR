import base64

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

width = -1
height = 256
ratio = 1.33
p_height = 165
p_width = 120
p_ratio = p_height/p_width
hashfunc = imagehash.dhash
threshold = 10
tolerance = 50

PRIVATE_KEY = "private"
PUBLIC_KEY = "public"

def extract_image(img, rect):
    rect = rect.astype(int)
    height = np.abs(rect[3, 1] - rect[0, 1])
    off_h = int((height * ratio - height) / 2)
    width = np.abs(rect[1, 0] - rect[0, 0])
    off_w = int((width * ratio - width) / 2)
    dims = img.shape
    # lim = np.array([max(rect[0, 1] - off_h, 0),
    #                 min(rect[3, 1] + off_h, dims[0]),
    #                 max(rect[0, 0] - int(width) - off_w, 0),
    #                 min(rect[0, 0] -  off_w, dims[1])])
    lim = np.array([max(rect[0, 1] - 2*off_h, 0), min(rect[3, 1] + 2*off_h, dims[0]),
                    max(rect[0, 0] - int(width) - 2*off_w, 0), min(rect[0, 0], dims[1])])
    lim = np.array(lim, dtype='uint16')
    region = img[lim[0]:lim[1],
             lim[2]:lim[3]]

    plt.figure(22)
    region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(region_gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    #thresh = cv2.threshold(sharpen, 100, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.threshold(sharpen,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    plt.imshow(close, 'gray')
    ax = plt.gca()
    dims = region.shape
    min_area = dims[0]*dims[1]*0.4
    max_area = dims[0]*dims[1]

    prev_area = 0
    ROI = region
    for c in cnts:
        print(".")
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            print("ok")
            x, y, w, h = cv2.boundingRect(c)
            if prev_area<area:
                prev_area = area
            ROI = region[y:y + h, x:x + w]
            p = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(p)

    return ROI, height, width

def compensate_perspective(img, rect):
    height = np.abs(rect[3, 1] - rect[0, 1])
    width = np.abs(rect[1, 0] - rect[0, 0])
    dim = int(max(height, width))
    srcRect = rect
    dstRect = np.float32([[srcRect[0,0],srcRect[0,1]],
                          [srcRect[0,0]+dim,srcRect[0,1]],
                          [srcRect[0,0]+dim,srcRect[0,1]+dim],
                          [srcRect[0,0],srcRect[0,1]+dim]])
    M = cv2.getPerspectiveTransform(srcRect, dstRect)
    dst = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
    dst = cv2.warpPerspective(img.astype(np.uint8), M, dst.shape)
    rect = dstRect

    return dst

def create_from_photo(photoRGB, privkey = None):
    width = p_width
    height = p_height
    photoResized = cv2.resize(photoRGB, (width, height))
    # evaluate hash from resized image
    hash = hashfunc(Image.fromarray(photoResized[3:-2,3:-2,:]))
    print("hash to be saved in QR = " + str(str(hash).encode()))
    rsa = PKCS1_OAEP.new(privkey)
    if privkey is None:
        hash_enc = hash
    else:
        hash_enc = rsa.encrypt(str(hash).encode())
    print(len(hash_enc))
    #print(rsa.encrypt(str(hash).encode(), privkey).hex())
    print(hash_enc)
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
    # Add red border
    photoResized[:,0:2] = np.array([0,255,0])
    photoResized[0:2, :] = np.array([0, 255, 0])
    photoResized[-2::, :] = np.array([0, 255, 0])
    photoResized[:, -2::] = np.array([0, 255, 0])
    dst[0:height, 0:width, :] = photoResized
    dst[0:height, width:width + height, :] = code_resized

    return dst, hash

def decrypt_message(msg, cipher):
    default_length = 128
    encrypt_byte = bytes(msg.decode(), 'iso_8859_1')
    length = len(encrypt_byte)
    #print(type(encrypt_byte))
    #print(length)
    #print(encrypt_byte)

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

def detect(privatekey = None, dev = 0, photo = None):
    cap = None
    if photo is None:
        cap = cv2.VideoCapture(dev)

        if cap is None or not cap.isOpened():
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
        _, img = cap.read()
    img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ax = plt.gca()
    im = ax.imshow(img_conv)

    while True:
        if photo is not None:
            img = photo
        else:
            _, img = cap.read()
        img_conv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        im.set_data(img_conv)
        #plt.draw()
        plt.pause(0.001)
        # detect and decode
        #data, bbox, _ = detector.detectAndDecode(img)
        ddd = pyzbar.decode(img)
        if len(ddd)==0:
            continue
        data = ddd[0].data
        bbox = ddd[0].polygon
        print(len(data.decode()))
        # check if there is a QRCode in the image
        if data:
            if (privatekey is not None):
                data = decrypt_message(data, cipher)
            else:
                data = data.decode()
            break

    # Release camera
    if cap is not None:
        cap.release()

    plt.subplot(1,2,1)
    plt.imshow(img_conv);
    plt.axis(False)
    hash_from_qr = imagehash.hex_to_hash(str(data))
    plt.title("Hash from QR: " + str(hash_from_qr))
    plt.plot(bbox[0].x, bbox[0].y, "r+")
    plt.plot(bbox[1].x, bbox[1].y, "b+")
    plt.plot(bbox[2].x, bbox[2].y, "g+")
    plt.plot(bbox[3].x, bbox[3].y, "y+")

    rect = np.array([[bbox[0].x,bbox[0].y],
                     [bbox[3].x,bbox[3].y],
                     [bbox[2].x,bbox[2].y],
                     [bbox[1].x,bbox[1].y]], 'float32')
    #
    plt.figure(2)
    compensated = cv2.cvtColor(compensate_perspective(img, rect), cv2.COLOR_BGR2RGB)
    plt.title("Compensated")
    plt.imshow(compensated)
    #

    plt.figure(1)
    plt.subplot(1, 2, 2)
    region, _, _ = extract_image(compensated, rect)
    plt.figure(1)
    # region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    plt.imshow(region)
    hash_from_image = hashfunc(Image.fromarray(region))
    print("Similarity = " + str(hash_from_image-hash_from_qr))
    plt.title(str(hash_from_image))
    plt.axis(False)
    dims = region.shape
    if hash_from_image-hash_from_qr <= threshold:
        result = cv2.imread("correct.png")
    else:
        result = cv2.imread("wrong.png")

    #print(dims)
    result = cv2.resize(result, (dims[1], dims[0]))
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result, alpha=0.3, interpolation='bilinear')
    plt.show()

def create(photo = None, privkey = None):
    if photo is None:
        photoName = sys.argv[2]
        photo = cv2.imread(photoName)
        photoRGB = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    else:
        photoRGB = photo
    dst, hash = create_from_photo(photoRGB, privkey)
    plt.figure(1)
    plt.imshow(dst)
    plt.axis(False)
    plt.title(str(hash))
    plt.show()

def acquire_video_until_button(dev = 0):
    cap = cv2.VideoCapture(dev)

    if cap is None or not cap.isOpened():
        print("Camera not found!")
        return None

    fig = plt.figure(1)
    _, img = cap.read()
    dims = img.shape
    dx = p_width*2
    dy = p_height*2
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
        p = plt.Rectangle((x,y), dx, dy, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(p)
        ret = plt.waitforbuttonpress(0.00001)
        if ret == True:
            break
    plt.ioff()
    cap.release()
    return img_conv[y:y+dy,x:x+dx,:]

def generate_keys():
    # Generate keys
    random_generator = Random.new().read
    key = RSA.generate(1024, random_generator)  # generate public and private keys

    # Save public key
    #print("public:  " + str(key.publickey().exportKey()))
    fid = open(PUBLIC_KEY, "wb")
    fid.write(key.publickey().export_key())
    fid.close()

    # Save private key
    #print("private: " + str(key.export_key()))
    fid = open(PRIVATE_KEY, "wb")
    fid.write(key.export_key())
    fid.close()
    return key

def read_keys():
    fid = open(PRIVATE_KEY, "r")
    privatekey = RSA.import_key(fid.read())
    fid.close()

    fid = open(PUBLIC_KEY, "r")
    publickey = RSA.import_key(fid.read())
    fid.close()

    return privatekey, publickey

if __name__ =="__main__":

    if sys.argv[1] == "create":
        img = None
        if len(sys.argv) < 3:
            img = acquire_video_until_button()
        create(img)
    elif sys.argv[1] == "createc":
        img = None
        key = generate_keys()
        if len(sys.argv) < 3:
            img = acquire_video_until_button()
        create(img, key.publickey())
    elif sys.argv[1] == "detect":
        if len(sys.argv)>2:
            detect(photo=cv2.imread(sys.argv[2]))
        else:
            detect()
    elif sys.argv[1] == "detectc":
        privateKey, publicKey = read_keys()
        if len(sys.argv)>2:
            detect(privateKey, photo=cv2.imread(sys.argv[2]))
        else:
            detect(privateKey)
