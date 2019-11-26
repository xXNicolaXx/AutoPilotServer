from keras.models import load_model
import socket
import sys
import datetime
import csv
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
model = load_model("../model/nvidiaModel.h5")
with open("../training/data/images/1574517989418963.jpg", "rb") as f:
    data = f.read()


def image_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))  # Image input size of the Nvidia model architecture
    image = (image / 127.5) - 1

    return image


image = Image.open(BytesIO(bytearray(data)))
image = np.asarray(image)
image = image_preprocess(image)
image = np.array([image])
steering_angle = str(model.predict_classes(image))
print(steering_angle)
