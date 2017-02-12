import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import \
    Dense, Dropout, Activation, Flatten, \
    Convolution2D, MaxPooling2D, Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

model_name = 'model'
data_path = './data/'
half_batch_size = 32
n_epochs = 30

def generate(xs, ys, half_batch_size):
    size = xs.shape[0]
    while True:
        xs, ys = shuffle(xs, ys)
        for begin in range(0, size, half_batch_size):
            out_xs = np.zeros([0, 20, 80, 3], dtype='uint8')
            out_ys = np.zeros(0, dtype='float32')
            for i in range(half_batch_size):
                img = xs[begin + i]
                angle = ys[begin + i]
                out_xs = np.concatenate([out_xs, [img]])
                out_ys = np.concatenate([out_ys, [angle]])
                out_xs = np.concatenate([out_xs, [np.fliplr(img)]])
                out_ys = np.concatenate([out_ys, [-angle]])
            yield(out_xs, out_ys)

def pred_steering(name='model'):
    model = Sequential()

    lambda0 = Lambda( lambda x: x/127.5 - 1.0, input_shape=(20,80,3) )
    model.add(lambda0)

    model.add(Convolution2D(8,5,3)) # out: 76,18
    model.add(Activation('elu'))
    model.add(Convolution2D(12,5,3)) # out: 72,16
    model.add(Activation('elu'))
    model.add(MaxPooling2D( pool_size=(1, 2), strides=(1,2)  )) # out 36,16

    model.add(Convolution2D(12,3,3)) # out: 34, 14
    model.add(Activation('elu'))
    model.add(Convolution2D(12,3,3)) # out: 32, 12
    model.add(Activation('elu'))
    model.add(MaxPooling2D()) # out: 16, 6

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('elu'))
    
    model.add(Dense(1))

    model_weights_file = name + '.h5'
    if os.path.exists(model_weights_file):
        model.load_weights(model_weights_file) 
        
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model


def save_model(model, name):
    model_file = name + '.json'
    model_weights_file = name + '.h5'

    if os.path.exists(model_file):
        os.remove(model_file)
    if os.path.exists(model_weights_file):
        os.remove(model_weights_file)

    model_json = model.to_json()

    with open(model_file, 'w') as file:
        file.write(model_json)
    
    model.save_weights(model_weights_file)


weight_save_callback = ModelCheckpoint('./weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
    monitor='val_loss', 
    verbose=0, save_best_only=True, mode='auto')


def main():
    xs = np.load('./xs.npy')
    ys = np.load('./ys.npy')

    xs, ys = shuffle(xs, ys)
    # x_train, x_valid, y_train, y_valid = train_test_split(xs, ys, test_size=0.1)

    rest = xs.shape[0] % half_batch_size
    xs = xs[:-rest]
    ys = ys[:-rest]

    model = pred_steering()
    history = model.fit_generator(generate(xs, ys, half_batch_size),
        samples_per_epoch=xs.shape[0] * 2,
        nb_epoch=n_epochs)

    save_model(model, model_name)

if __name__ == '__main__':
    main()