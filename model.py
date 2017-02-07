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
import os

model_name = 'model'
data_path = './data/'
df = pd.read_csv(data_path + 'driving_log.csv')
batch_size = 64
samples_epoch = batch_size * 200
n_epochs = 50


def generate_samples(df, batch_size, data_path):
    
    size = df.iloc[:,3].size
    offset = 0.23
    
    n_bin = 100
    
    vmin = np.absolute(df.values[:,3]).min()
    vmax = np.absolute(df.values[:,3]).max()
    
    bins = np.linspace(vmin, vmax, num=n_bin+1)
    bin_inds = np.digitize(df[[3]].values, bins, right= True)

    while True:
        imgs = np.zeros([0, 80, 160, 1], dtype='uint8')
        steerings = np.zeros(0, dtype='float32')

        for i in range(batch_size):
            # choose a bin
            upper_bound = np.random.choice(np.arange(1,n_bin + 1))
            indices = df[bin_inds == upper_bound].index

            while indices.size == 0:
                upper_bound = np.random.choice(np.arange(1,n_bin + 1))
                indices = df[bin_inds == upper_bound].index
            
            index_begin = np.random.choice(indices)            

            camera = np.random.choice([0,1,2])

            flip = np.random.randint(2)
            
            
            file_path = df.iloc[index_begin, camera]
            _, _, filename = file_path.partition('IMG')

            img = cv2.imread(data_path + 'IMG' + filename)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[60:140 ,:]
            img = cv2.resize(img, (160, 80))
            img = img[...,None]

            steering = df.iloc[index_begin, 3]

            if camera == 1:
                steering += offset
            elif camera == 2:
                steering -= offset

            if flip:
                img = np.fliplr(img)
                steering = -steering

            imgs = np.append(imgs, [img], axis=0)
            steerings = np.append(steerings, [steering], axis=0)

        yield imgs, steerings


def pred_steering():
    model = Sequential()

    lambda0 = Lambda( lambda x: x/127.5 - 1.0, input_shape=(80,160,1) )
    model.add(lambda0)
    
    model.add(Convolution2D(16,3,5,border_mode='valid',subsample=(1,2)))
    model.add(Activation('elu'))  #out (78, 78)

    model.add(Convolution2D(16,3,3,border_mode='valid'))
    model.add(Activation('elu'))  #out (76, 76)
    model.add(MaxPooling2D(pool_size=(2, 2))) # out (38, 38)
    
    model.add(Convolution2D(24,3,3,border_mode='valid'))
    model.add(Activation('elu'))  # out (36,36)
    model.add(MaxPooling2D(pool_size=(2, 2))) # out (18, 18)
    
    model.add(Convolution2D(32,3,3,border_mode='valid'))
    model.add(Activation('elu')) # out (16, 16)
    model.add(MaxPooling2D(pool_size=(2, 2))) # output (8, 8)
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(32,3,3,border_mode='valid'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # output (3, 3)
    model.add(Dropout(0.5))

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


weight_save_callback = ModelCheckpoint('./weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
    monitor='val_loss', 
    verbose=0, save_best_only=True, mode='auto')



def main():

    model = pred_steering()

    history = model.fit_generator(generate_samples(df, batch_size, data_path),
        samples_per_epoch=samples_epoch,
        nb_epoch=n_epochs,
        callbacks=[])

    save_model(model, model_name)

if __name__ == '__main__':
    main()