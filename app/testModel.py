import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

modelN = tf.keras.models.load_model('../model/nvidiaModel.h5')
with open("../data/images/1574517989418963.jpg", "rb") as f:
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
steering_angle = str(modelN.predict_classes(image))
print(steering_angle)
