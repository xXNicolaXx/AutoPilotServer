import socket
import sys
import datetime
import csv
from keras.models import load_model
import cv2
from PIL import Image
from io import BytesIO
import numpy as np

HOST = str(sys.argv[1])  # this is your localhost
PORT = int(sys.argv[2])
MODE = str(sys.argv[3])
print("Mode:", MODE)


model = load_model("../model/nvidiaModel_long_train.h5")


def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# socket.socket: must use to create a socket.
# socket.AF_INET: Address Format, Internet = IP Addresses.
# socket.SOCK_STREAM: two-way, connection-based byte streams.
print('socket created')

# Bind socket to Host and Port
try:
    s.bind((HOST, PORT))
except socket.error as err:
    print('Bind Failed, Error Code: ' + str(err))
    sys.exit()

print('Socket Bind Success!')
# listen(): This method sets up and start TCP listener.
s.listen(10)
print('Socket is now listening')
conn, addr = s.accept()
print('Connect with ' + addr[0] + ':' + str(addr[1]))
data = b''
imageArray = []
dateArray = []
directionArray = []

if MODE == "TRAINING":

    direction = "4"
    drive = True
    # TRAINING MODE
    while drive:
        buf = conn.recv(8192)
        if buf[-3:] != b'eof':
            data = data + buf
        else:
            data = data + buf[:-4]
            direction = buf[-4:len(buf) - 3].decode('utf-8')
            directionArray.append(direction)
            imageArray.append(data)
            date_string = str(datetime.datetime.now().timestamp()).replace('.', '')
            dateArray.append(date_string)
            data = b''
        if direction == "9":
            break
    print("Saving collected images")
    for index, item in enumerate(imageArray):
        with open('./images/' + dateArray[index] + '.jpg', 'wb') as f:
            f.write(item)
    print("Saving data to csv")
    with open('data.csv', 'w') as writeFile:
        writer = csv.writer(writeFile, delimiter=',', quotechar='|')
        for index, item in enumerate(dateArray):
            writer.writerow(['images/' + item + '.jpg', str(directionArray[index])])

    print("Finish")

else:

    # AI MODE
    while True:
        buf = conn.recv(8192)

        if buf[-3:] != b'eof':
            data = data + buf
        else:
            data = data + buf[:-3]

            image = Image.open(BytesIO(bytearray(data)))
            image = np.asarray(image)
            image = img_preprocess(image)
            image = np.array([image])
            steering_angle = str(model.predict_classes(image))
            steering_angle = steering_angle[1:-1]
            conn.send(str.encode('{0}\n'.format(steering_angle)))
            if steering_angle == "0":
                print("left")
            elif steering_angle == "1":
                print("forward")
            elif steering_angle == "2":
                print("right")
            data = b''
