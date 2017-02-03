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

data_file = './data/'
df = pd.read_csv(data_file + 'driving_log.csv')

cv2.imread

def sample(batch_size):
    
    while True:
        begin = np.random.randint(0, df.steering.size - batch_size)
        end = begin + batch_size
        samples = df.ix[range(begin, end), range(4)]
        if np.absolute(samples.steering).sum() > batch_size * 0.1:
            return samples, begin


def generate_samples(batch_size):
        while True:
            df_samples, index_begin = sample(batch_size)

            imgs = np.zeros([0, 160, 320, 3])

            for i in range(batch_size):
                file_path = df_samples.center[index_begin + i]
                img = plt.imread(data_file + file_path)
                imgs = np.concatenate([imgs, [img]])    
            yield imgs, df_samples.steering

def pred_steering():
    model = Sequential()
    model.add(Convolution2D(32,3,3,border_mode='same',subsample=(2,2),\
        input_shape=(160,320,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3,border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # output (32, 40, 80)
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model



def main():
    model = pred_steering()
    history = model.fit_generator(generate_samples(16),samples_per_epoch=100,nb_epoch=1)


if __name__ == '__main__':
    main()