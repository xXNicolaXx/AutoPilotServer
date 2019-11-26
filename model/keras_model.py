import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters
import cv2
import pandas as pd
import ntpath
import random
import time
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
datadir = "../training/data/csv/"
imagedir = "../training/data/images/"
columns = ["image", "direction"]
data = pd.read_csv(os.path.join(datadir, "data.csv"), names=columns)
pd.set_option("display.max_colwidth", -1)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
num_bins = 3
samples_per_bin = 400


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def show_all_data():
    print(data.describe())
    print(data.head())


def cut_image_path():
    data["center"] = imagedir + data["center"].apply(path_leaf)
    data["left"] = imagedir + data["left"].apply(path_leaf)
    data["right"] = imagedir + data["right"].apply(path_leaf)


def show_initial_steering_data():
    i_hist, i_bins = np.histogram(data["direction"], num_bins)
    center = (i_bins[:-1] + i_bins[1:]) * 0.5
    plt.bar(center, i_hist, width=0.05)
    plt.title("Direction data")
    plt.xlabel("Direction")
    plt.ylabel("Number of data")
    plt.plot((np.min(data["direction"]), np.max(data["direction"])), (samples_per_bin, samples_per_bin))
    plt.show()


show_all_data()
show_initial_steering_data()

hist, bins = np.histogram(data["direction"], num_bins)
remove_list = []
for i in range(num_bins):
    list_ = []
    for j in range(len(data["direction"])):
        if bins[i] <= data["direction"][j] <= bins[i + 1]:
            list_.append(j)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

data.drop(data.index[remove_list], inplace=True)


def show_modified_steering_data():
    m_hist, _ = np.histogram(data['direction'], num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5
    plt.bar(center, m_hist, width=0.05)
    plt.plot((np.min(data['direction']), np.max(data['v'])), (samples_per_bin, samples_per_bin))
    plt.show()


# show_modified_steering_data()


def load_data():
    image_paths = data["image"].values
    steerings = data["direction"].values

    return train_test_split(image_paths, steerings, test_size=0.2, random_state=5)


def zoom_image(image):
    zoom = augmenters.Affine(scale=(1, 1.3))
    return zoom.augment_image(image)


def pan_image(image):
    pan = augmenters.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    return pan.augment_image(image)


def image_brightness(image):
    brightness = augmenters.Multiply((0.1, 2))
    return brightness.augment_image(image)


def flip_image(image, steering_angle):
    # We need to "flip" also the steering angle as the image as flipped

    image = cv2.flip(image, 1)
    steering_angle = steering_angle[::-1]

    return image, steering_angle


def augment_image(image, steering_angle):
    image = mpimg.imread("../training/data/" + image)
    if np.random.rand() < 0.5:
        image = pan_image(image)
    if np.random.rand() < 0.5:
        image = zoom_image(image)
    if np.random.rand() < 0.5:
        image = image_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = flip_image(image, steering_angle)

    return image, steering_angle


def image_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))  # Image input size of the Nvidia model architecture
    image = (image / 127.5) - 1

    return image


def batch_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        batch_image = []
        batch_steering = []
        for _ in range(batch_size):
            index = random.randint(0, len(image_paths) - 1)

            image = image_paths[index]

            steering_angle = steering_angles[index]
            if is_training:

                image, steering_angle = augment_image(image, steering_angle)
            else:
                image = mpimg.imread("../training/data/" + image)
                steering_angle = steering_angle

            image = image_preprocess(image)
            batch_image.append(image)
            batch_steering.append(steering_angle)
        yield (np.asarray(batch_image), np.asarray(batch_steering))


def nvidia_model():
    import keras
    """
        NVIDIA model used
        Image normalization to avoid saturation and make gradients work better.
        Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Drop out (0.5)
        Fully connected: neurons: 100, activation: ELU
        Fully connected: neurons: 50, activation: ELU
        Fully connected: neurons: 10, activation: ELU
        Fully connected: neurons: 1 (output)
        # the convolution layers are meant to handle feature engineering
        the fully connected layer for predicting the steering angle.
        dropout avoids overfitting
        ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
        """

    model = Sequential()
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="elu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="elu"))
    model.add(Flatten())

    model.add(Dense(units=100, activation="elu"))
    model.add(Dense(units=50, activation="elu"))
    model.add(Dense(units=10, activation="elu"))
    model.add(Dense(units=3, activation='softmax'))

    optimizer = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    return model


nvidia_model = nvidia_model()
print(nvidia_model.summary())
es = EarlyStopping(monitor='acc', mode='max', verbose=1)
start_time = time.clock()
# cut_image_path()
X_train, X_valid, y_train, y_valid = load_data()
y_train = to_categorical(y_train, 3)
y_valid = to_categorical(y_valid, 3)
history = nvidia_model.fit_generator(batch_generator(X_train, y_train, 32, True),
                                     steps_per_epoch=len(X_train),
                                     epochs=30,
                                     validation_data=batch_generator(X_valid, y_valid, 32, False),
                                     validation_steps=200,
                                     callbacks=[es],
                                     verbose=True,
                                     shuffle=True)

print("--- trained in %s seconds ---" % (time.clock() - start_time))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
nvidia_model.save('nvidiaModel.h5')
