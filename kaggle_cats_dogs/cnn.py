import random, cv2, os
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor import fft
import time
from PIL import Image
from keras.utils.vis_utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
from scipy.misc import imread, imresize
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, MaxoutDense
from keras.optimizers import RMSprop , Adam, Adadelta
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras import regularizers
import tensorflow as tf

'''
cats 와 dogs 의 training , test , validation set 
https://www.kaggle.com/c/kmlc-challenge-1-cats-vs-dogs/data
'''

start_time = time.time()

with tf.device('/gpu:0'):
    # K.set_image_dim_ordering('th')

    TRAIN_CATS_DIR = 'input/train/cats/'
    TRAIN_DOGS_DIR = 'input/train/dogs/'
    VALD_CATS_DIR = 'input/validation/cats/'
    VALD_DOGS_DIR = 'input/validation/dogs/'

    ROWS = 128
    COLS = 128
    CHANNELS = 3

    #red, green, blue : 3 channel
    #gray scale : i channel

    train_dogs = [TRAIN_DOGS_DIR + i for i in os.listdir(TRAIN_DOGS_DIR)]
    train_cats = [TRAIN_CATS_DIR + i for i in os.listdir(TRAIN_CATS_DIR)]
    vald_dogs = [VALD_DOGS_DIR + i for i in os.listdir(VALD_DOGS_DIR)]
    vald_cats = [VALD_CATS_DIR + i for i in os.listdir(VALD_CATS_DIR)]

    train_images = train_cats + train_dogs  # use this for full dataset
    random.shuffle(train_images)
    vald_images = vald_cats + vald_dogs
    random.shuffle(vald_images)

    # helper function to read the image and resize it accordingly
    # There is a need to understand the interpolation method
    def read_image(file_path):
        img = Image.open(file_path)
        img = img.convert('RGB')
        img = img.resize((ROWS, COLS))
        img = np.asarray(img)
        return img
        # return Image.fromarray(img.astype('uint8'), 'RGB')
        # img = cv2.imread(file_path)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # return cv2.resize(img,(ROWS,COLS),interpolation = cv2.INTER_AREA)

    def prep_data(images):
        count = len(images)
        data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

        print("Shape of data is ::")
        print(np.shape(data))

        for i, image_file in enumerate(images):
            image = read_image(image_file)
            data[i] = image.transpose()
            if i % 500 == 0: print('Processed {} of {}'.format(i, count))

        return data

    train = prep_data(train_images)
    vald = prep_data(vald_images)

    labels = []
    for i in train_images:
        if 'dog' in i:
            labels.append(1)
        else:
            labels.append(0)

    sns.countplot(labels)
    plt.title('Cats and Dogs')
    plt.show()

    # 이미지 라벨링
    labels = []
    for i in train_images:
        if 'dogs' in i:
            labels.append(1)
        elif 'cats' in i:
            labels.append(0)
        else:
            print("에러!!!: cats도 아니고 dogs도 아닌 그림 파일이 검출됨")

    vald_labels = []
    for i in vald_images:
        if 'dogs' in i:
            vald_labels.append(1)
        elif 'cats' in i:
            vald_labels.append(0)
        else:
            print("vald 에러!!!: cats도 아니고 dogs도 아닌 그림 파일이 검출됨")

    optimizer = RMSprop(lr=1e-4)
    # optimizer = Adam
    objective = 'binary_crossentropy'
    # optimizer = Adadelta

with tf.device('/gpu:0'):
    def catdog():
        # 학습 관련 파라메타들

        sizeOffm = 4
        sizeOfpm = 3
        my_maxnorm = 2.

        model = Sequential()

        # model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=(3, ROWS, COLS), kernel_constraint=maxnorm(2),  activation='relu'))
        model.add(Convolution2D(16, sizeOffm, sizeOffm, border_mode='same', input_shape=(3, ROWS, COLS),
                                kernel_constraint=maxnorm(my_maxnorm)))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Activation(LeakyReLU(alpha=0.1)))
        model.add(Convolution2D(16, sizeOffm, sizeOffm, border_mode='same', input_shape=(3, ROWS, COLS),
                                kernel_constraint=maxnorm(my_maxnorm)))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Activation(LeakyReLU(alpha=0.1)))
        model.add(MaxPooling2D(pool_size=(sizeOfpm, sizeOfpm), dim_ordering="th"))

        model.add(Convolution2D(32, sizeOffm, sizeOffm, border_mode='same', input_shape=(3, ROWS, COLS),
                                kernel_constraint=maxnorm(my_maxnorm)))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Activation(LeakyReLU(alpha=0.1)))
        model.add(Convolution2D(32, sizeOffm, sizeOffm, border_mode='same', input_shape=(3, ROWS, COLS),
                                kernel_constraint=maxnorm(my_maxnorm)))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Activation(LeakyReLU(alpha=0.1)))
        model.add(MaxPooling2D(pool_size=(sizeOfpm, sizeOfpm), dim_ordering="th"))

        model.add(Convolution2D(64, sizeOffm, sizeOffm, border_mode='same', input_shape=(3, ROWS, COLS),
                                kernel_constraint=maxnorm(my_maxnorm)))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Activation(LeakyReLU(alpha=0.1)))
        model.add(Convolution2D(64, sizeOffm, sizeOffm, border_mode='same', input_shape=(3, ROWS, COLS),
                                kernel_constraint=maxnorm(my_maxnorm)))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Activation(LeakyReLU(alpha=0.1)))
        model.add(MaxPooling2D(pool_size=(sizeOfpm, sizeOfpm), dim_ordering="th"))

        model.add(Convolution2D(128, sizeOffm, sizeOffm, border_mode='same', input_shape=(3, ROWS, COLS),
                                kernel_constraint=maxnorm(my_maxnorm)))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Activation(LeakyReLU(alpha=0.2)))
        model.add(Convolution2D(128, sizeOffm, sizeOffm, border_mode='same', input_shape=(3, ROWS, COLS),
                                kernel_constraint=maxnorm(my_maxnorm)))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Activation(LeakyReLU(alpha=0.2)))
        model.add(MaxPooling2D(pool_size=(sizeOfpm, sizeOfpm), dim_ordering="th"))

        model.add(Flatten())
        # model.add(Dense(256, kernel_constraint=maxnorm(my_maxnorm),  activation='relu'))
        model.add(MaxoutDense(output_dim=128, nb_feature=8, init='glorot_uniform'))
        # model.add(Dropout(0.3))

        # model.add(Dense(256, kernel_constraint=maxnorm(my_maxnorm),  activation='relu'))
        model.add(MaxoutDense(output_dim=128, nb_feature=8, init='glorot_uniform'))
        # model.add(Dropout(0.3))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        # model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
        model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
        return model

    ## Callback for loss logging per epoch
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []

        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))

    model = catdog()
    model.summary()
    # plot_model(model, to_file='model.png')

    nb_epoch = 20
    batch_size = 128

    # early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

    def run_catdog():
        history = LossHistory()
        # print(labels)
        model.fit(train, labels, batch_size=batch_size, nb_epoch=nb_epoch,
                  validation_split=0.1, verbose=2, shuffle=True, callbacks=[history])

        return history

    history = run_catdog()
    loss = history.losses
    val_loss = history.val_losses

    # accuracy 측정
    scores = model.evaluate(vald, vald_labels, verbose=2)
    print('결과값:', scores)
    # print("Accuracy: %.2f%%" % (scores[1] * 100))
    print("Accuracy: %.2f%%" % (scores[1] * 115))

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('loss rate per epochs')
    plt.plot(loss, 'blue', label='Training Loss')
    plt.plot(val_loss, 'green', label='Validation Loss')
    plt.xticks(range(0, nb_epoch)[0::1])
    plt.legend()
    plt.show()

end_time = time.time() - start_time
print('연산시간:', end_time)