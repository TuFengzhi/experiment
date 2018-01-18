from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.np_utils import to_categorical

from configs import bcolors

import cv2
import numpy as np

brightness = [-90, -60, -30, 0, 30, 60, 90]
noise = ['gauss']

def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch= image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)        
        noisy = image + image * gauss
        return noisy

def Model1(input_tensor=None, train=False):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)

    if train:
        batch_size = 256
        nb_epoch = 20

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        # add more train data
        x_train_new = []
        y_train_new = []
        for item in x_train:
            for ratio in brightness:
                img = item + ratio
                img[img < 0] = 0
                img[img > 255] = 255
                x_train_new.append(img)
        for item in y_train:
            for ratio in brightness:
                y_train_new.append(item)
        for item in x_train:
            for ratio in noise:
                x_train_new.append(noisy(ratio, item))
        for item in y_train:
            for ratio in noise:
                y_train_new.append(item)
        x_train_new = np.array(x_train_new)
        y_train_new = np.array(y_train_new)

        input_shape = (img_rows, img_cols, 1)

        x_train_new = x_train_new.astype('float32')
        x_test = x_test.astype('float32')
        x_train_new /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train_new = to_categorical(y_train_new, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print(bcolors.FAIL + 'you have to proved input_tensor when testing')
        exit()

    # block1
    x = Convolution2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(12, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # trainig
        model.fit(x_train_new, y_train_new, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model
        model.save_weights('./Models/Model1_new.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else:
        model.load_weights('./Models/Model1_new.h5')
        print(bcolors.OKBLUE + 'New Model1 loaded' + bcolors.ENDC)

    return model

def Model2(input_tensor=None, train=False):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)

    if train:
        batch_size = 256
        nb_epoch = 20

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        # add more train data
        x_train_new = []
        y_train_new = []
        for item in x_train:
            for ratio in brightness:
                img = item + ratio
                img[img < 0] = 0
                img[img > 255] = 255
                x_train_new.append(img)
        for item in y_train:
            for ratio in brightness:
                y_train_new.append(item)
        for item in x_train:
            for ratio in noise:
                x_train_new.append(noisy(ratio, item))
        for item in y_train:
            for ratio in noise:
                y_train_new.append(item)
        x_train_new = np.array(x_train_new)
        y_train_new = np.array(y_train_new)

        input_shape = (img_rows, img_cols, 1)

        x_train_new = x_train_new.astype('float32')
        x_test = x_test.astype('float32')
        x_train_new /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train_new = to_categorical(y_train_new, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print(bcolors.FAIL + 'you have to proved input_tensor when testing')
        exit()

    # block1
    x = Convolution2D(9, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(24, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(84, activation='relu', name='fc1')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # trainig
        model.fit(x_train_new, y_train_new, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model
        model.save_weights('./Models/Model2_new.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else:
        model.load_weights('./Models/Model2_new.h5')
        print(bcolors.OKBLUE + 'New Model2 loaded' + bcolors.ENDC)

    return model

def Model3(input_tensor=None, train=False):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)

    if train:
        batch_size = 256
        nb_epoch = 20

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        # add more train data
        x_train_new = []
        y_train_new = []
        for item in x_train:
            for ratio in brightness:
                img = item + ratio
                img[img < 0] = 0
                img[img > 255] = 255
                x_train_new.append(img)
        for item in y_train:
            for ratio in brightness:
                y_train_new.append(item)
        for item in x_train:
            for ratio in noise:
                x_train_new.append(noisy(ratio, item))
        for item in y_train:
            for ratio in noise:
                y_train_new.append(item)
        x_train_new = np.array(x_train_new)
        y_train_new = np.array(y_train_new)

        input_shape = (img_rows, img_cols, 1)

        x_train_new = x_train_new.astype('float32')
        x_test = x_test.astype('float32')
        x_train_new /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train_new = to_categorical(y_train_new, nb_classes)
        y_test = to_categorical(y_test, nb_classes)

        input_tensor = Input(shape=input_shape)
    elif input_tensor is None:
        print(bcolors.FAIL + 'you have to proved input_tensor when testing')
        exit()

    # block1
    x = Convolution2D(9, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(24, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # trainig
        model.fit(x_train_new, y_train_new, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        # save model
        model.save_weights('./Models/Model3_new.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
    else:
        model.load_weights('./Models/Model3_new.h5')
        print(bcolors.OKBLUE + 'New Model3 loaded' + bcolors.ENDC)

    return model

if __name__ == '__main__':
    print('Start Retraining Model1')
    Model1(train=True)
    print('Start Retraining Model2')
    Model2(train=True)
    print('Start Retraining Model3')
    Model3(train=True)
