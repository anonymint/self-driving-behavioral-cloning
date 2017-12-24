import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_data(data_locations):
    """
    Load data from array of data_lications then correct the path to images before saving and return

    Features: Center Camera only
    Labels: Steering angle

    :param data_locations:
    :return:
    """
    lines = []
    for loc in data_locations:
        with open(loc + '/driving_log.csv', 'r') as csvf:
            reader = csv.reader(csvf)
            for l in reader:
                l[0] = loc + '/IMG/' + l[0].split('/')[-1]
                l[1] = loc + '/IMG/' + l[1].split('/')[-1]
                l[2] = loc + '/IMG/' + l[2].split('/')[-1]
                lines.append(l)
    return lines


def load_data_normalize(data_locations):
    """
    Load data from array of data_lications then correct the path to images before saving and return
    In addition, apply probability to filter out

    0.0 steering angle to overcome bias going straight
    -1.0 and 1.0 steeting angle so we don't have outlier data at far end

    Features: Center Camera only
    Labels: Steering angle

    :param data_locations:
    :return:
    """
    lines = []
    for loc in data_locations:
        with open(loc + '/driving_log.csv', 'r') as csvf:
            reader = csv.reader(csvf)
            for l in reader:
                random_prob = random.random()
                angel = float(l[3])
                if (angel == 0.0 and random_prob <= 0.3) or (abs(angel) == 1.0 and random_prob <= 0.1) or (
                        abs(angel) > 0.0 and abs(angel) < 1.0):
                    l[0] = loc + '/IMG/' + l[0].split('/')[-1]
                    l[1] = loc + '/IMG/' + l[1].split('/')[-1]
                    l[2] = loc + '/IMG/' + l[2].split('/')[-1]
                    lines.append(l)
    return lines


def visualize_data(lines):
    """
    Visualize data and show random captured images

    :param lines:
    :return:
    """
    print('size of data', len(lines))
    n = random.randint(0, len(lines))
    fig = plt.figure(figsize=(20, 200))
    for i in range(3):
        image = cv2.cvtColor(cv2.imread(lines[n][i]), cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(image)


def visualize_histogram(lines, bucket_size = 10):
    """
    Visualize data in bucket_size, this is useful to visualize data

    :param lines:
    :param bucket_size:
    :return:
    """
    plt.figure(figsize=(10, 10))
    measure = [float(l[3]) for l in lines]
    plt.title('Histogram of steering angle')
    plt.hist(measure, bucket_size, color='green')


def get_data(data_root_loc):
    """
    Load all data in memory

    Features: Center Camera only
    Labels: Steering angle

    :param data_root_loc: array of input data location
    :return:
    """
    with open(data_root_loc + '/driving_log.csv', 'r') as csvf:
        reader = csv.reader(csvf)
        lines = [l for l in reader]

    images = []
    steering_angle = []

    for line in lines:
        current_image_location = data_root_loc + '/IMG/'
        center_f = line[0].split('/')[-1]
        steering_measurement = float(line[3])
        # center double data with flip
        center_image = cv2.imread(current_image_location + center_f)
        center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        images.append(center_image)
        images.append(np.fliplr(center_image))
        steering_angle.append(steering_measurement)
        steering_angle.append(-steering_measurement)

    return np.array(images), np.array(steering_angle)

def generator(samples, batch_size=32):
    """
    Data generator please be noted it double the size of samples with flip and angle adjustment

    :param samples:
    :param batch_size:
    :return:
    """
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering_angle = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                # convert BGR to RGB because cv2 read in as BGR
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                steering_measurement = float(batch_sample[3])
                # add data in with flip technique
                images.append(center_image)
                images.append(np.fliplr(center_image))
                steering_angle.append(steering_measurement)
                steering_angle.append(-steering_measurement)
            x_train = np.array(images)
            y_train = np.array(steering_angle)
            yield shuffle(x_train, y_train)


def simple_network(input_shape):
    """
    Very simple network

    :param input_shape:
    :return:
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1))
    return model


def lenet_network(input_shape):
    """
    LeNet architecture

    :param input_shape:
    :return:
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(12, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
    return model


def nvidia_network(input_shape):
    """
    Nvidia architecture

    :param input_shape:
    :return:
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def train(x_train, y_train, model, saved_name='model.h5'):
    """
    Train will all data of x and y in memory, good to try something quick or prototype please avoid using it
    when data is so huge

    :param x_train:
    :param y_train:
    :param model:
    :param saved_name:
    :return:
    """
    model.compile(loss='mse', optimizer='adam')
    history_data = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=30, verbose=1)
    model.save(saved_name)
    return history_data


def train_with_generator(train_generator, validation_generator, train_size, validation_size, epoch, model,
                         saved_name='model.h5'):
    """
    Model training with generation which is much more efficient than normal fit since we don't put all data in memory
    at once but rather a chunk of data please see train function for equivalent

    :param train_generator:
    :param validation_generator:
    :param train_size:
    :param validation_size:
    :param epoch:
    :param model:
    :param saved_name:
    :return:
    """
    model.compile(loss='mse', optimizer='adam')
    history_data = model.fit_generator(train_generator, samples_per_epoch=train_size,
                                       validation_data=validation_generator, nb_val_samples=validation_size,
                                       nb_epoch=epoch, verbose=1)
    model.save(saved_name)
    return history_data


def visualize_loss(history):
    """
    Plot model history of error loss of training and validation

    :param history: keras history
    :return:
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    # load data from 3 location data_v3, curve_drive and side_drive
    data = load_data_normalize(['./data_v3', './curve_drive', './side_drive'])

    # Split data to train and validation 80/20 ratio
    train_samples, validation_samples = train_test_split(data, test_size=0.2)

    # create Train and Validation generator which will be fetched to model training later on
    train_generator = generator(train_samples, batch_size=16)
    validation_generator = generator(validation_samples, batch_size=16)

    # Model
    shape_of_data = (160, 320, 3)
    train_model = nvidia_network(shape_of_data)

    # Training model
    # The size of train and validation are double because at generator I add flipping images for every images.
    history_data = train_with_generator(train_generator, validation_generator, len(train_samples) * 2,
                                        len(validation_samples) * 2, epoch=5, model=train_model)
