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
import os

model_name = 'model'
data_path = './data/'
df = pd.read_csv(data_path + 'driving_log.csv')
<<<<<<< HEAD
batch_size = 64
samples_epoch = batch_size * 300
n_epochs = 20
=======
batch_size = 32
samples_epoch = batch_size * 250
n_epochs = 10
>>>>>>> 1103c3d7deda58596f6a86972dafa12088c4d427


def generate_samples(batch_size):
    means = np.zeros(0)
    size = df.steering.size
    offset = 0.125

    n_bin = 5
    
    for i in range(0, size - batch_size):
        samples = df.ix[range(i, i + batch_size), range(n_bin + 1)]
        mean = np.absolute( samples.steering ).mean()
        means = np.append(means, [mean])

    bins = np.linspace(means.min(), means.max(), num=8)
    bin_inds = np.digitize(means, bins, right= True)
    bin_inds = np.append(bin_inds, np.zeros(df.shape[0] - bin_inds.size))

    while True:
        # choose a bin
        upper_bound = np.random.choice(np.arange(1,n_bin + 1), p=[0.1,0.1,0.15,0.15,0.5])
        indices = df.iloc[bin_inds == upper_bound].index
        # choose a sequence from the bin
        index_begin = np.random.choice(indices)

        imgs = np.zeros([0, 160, 320, 3], dtype='uint8')
        steerings = np.zeros(0)

        camera = 'center'
        if upper_bound == n_bin:
            camera_selection = np.random.randint(3)
            if camera_selection == 1:
                camera = 'left'
            elif camera_selection == 2:
                camera = 'right'

        flip = np.random.randint(2)

        for i in range(batch_size):
            file_path = df[camera][index_begin + i]
            file_path = file_path.strip()
            
            img = plt.imread(data_path + file_path)
            
            steering = df.steering[index_begin + i]
            if camera == 'left':
                steering += offset
            elif camera == 'right':
                steering -= offset

            if flip:
                img = np.fliplr(img)
                steering = -steering

            imgs = np.append(imgs, [img], axis=0)
            steerings = np.append(steerings, [steering], axis=0)

        yield imgs, steerings

def pred_steering():
    model = Sequential()

    lambda0 = Lambda( lambda x: x/127.5 - 1., input_shape=(160,320,3) )
    model.add(lambda0)
<<<<<<< HEAD
    
    model.add(Convolution2D(16,4,4,border_mode='same',subsample=(4,4)))
    model.add(Activation('relu')) # output (40, 80)
    model.add(Convolution2D(32,3,3,border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # output (20, 40)
    
=======
    model.add(Convolution2D(8,4,4,border_mode='same',subsample=(4,4)))
    model.add(Activation('relu')) # output (8, 40, 80)
    model.add(Convolution2D(16,3,3,border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # output (16, 20, 40)

    model.add(Convolution2D(32,2,2,border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # output (32, 10, 20)

>>>>>>> 1103c3d7deda58596f6a86972dafa12088c4d427
    model.add(Convolution2D(64,2,2,border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # output (10, 20)
    
    model.add(Convolution2D(128,2,2,border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
<<<<<<< HEAD
    model.add(MaxPooling2D(pool_size=(2, 2))) # output (5, 10)

    model.add(Convolution2D(256,2,2,border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same')) # output (3, 5)
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
=======
    model.add(MaxPooling2D(pool_size=(2, 2))) # output (64, 5, 10)

    model.add(Convolution2D(128,2,2,border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same')) # output (128, 3, 5)
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
>>>>>>> 1103c3d7deda58596f6a86972dafa12088c4d427
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model


def save_model(model, name):
    model_file = name + '.json'
    model_weights_file = name + '.h5'

    if os.path.exists(model_file):
        os.remove(model_file)
    if os.path.exists(model_weights_file):
        os.remove(model_weights_file)

    model_json = model. to_json()

    with open(model_file, 'w') as file:
        file.write(model_json)
    
    model.save_weights(model_weights_file)


def main():
    model = pred_steering()
    history = model.fit_generator(generate_samples(batch_size),samples_per_epoch=samples_epoch,nb_epoch=n_epochs)
    save_model(model, model_name)

if __name__ == '__main__':
    main()