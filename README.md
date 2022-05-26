# photoQR: A novel ID card with an encoded view
There is an increasing interest in developing techniques to identify and assess data to allow an easy
and continuous access to resources, services or places that require thorough ID control. Usually, in
order to give access to these resources, different kinds of documents are mandatory. In order to
avoid forgeries without the need of extra credentials, a new system –named photoQR, is here
proposed. This system is based on a ID card having two objects: one person’s picture (pre-processed
via blur and/or swirl techniques) and one QR code containing embedded data related to the picture.
The idea is that the picture and the QR code can assess each other by a proper hash value in the QR.
The QR without the picture cannot be assessed and vice versa. An open source prototype of the
photoQR system has been implemented in Python and can be used both in offline and real-time
environments, which effectively combines security concepts and image processing algorithms to
obtain data assessment.

Run the prototype application:
  user:~$ git clone https://github.com/canessae/photoQR.git
  user:~$ cd photoQR/python
This command downloads the Python source code. In order to have a clean installation, it is
suggested to use a Python virtual environment:
  user:~$ python3 -m venv venv
  user:~$ source venv/bin/activate
Now it is necessary to fulfill all requirements which can be simply downloaded with the following
command:
  user:~$ pip3 install -r requirements.txt
All needed dependencies of the prototype are stored in the requirements.txt file, and the prototype is
ready to be used. The user can create a new photoQR ID card running the command:
  user:~$ python3 testPhotoQR.py create
This command tries to open the camera and wait for a button to take a photo which will be used for
the card creation. If the user specifies an image filename after the “create” command, the photoQR
ID card is generated using that picture. The detection algorithm can be used, as for the creation,
using the camera or a specified image file.
  user:~$ python3 testPhotoQR.py detect <Figure_1.png>
If nothing is specified the code tries to open the camera and wait for a button press in order to take a
photo and to perform the card assessment. If an image filename is specified after the “detect”
command, that image is used for assessment (Figure_1.png in the example above).
