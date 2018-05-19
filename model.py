import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

#prefix = './ds_udacity/data'
prefix = './ds1'
lines = []
with open(prefix + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

nrows,ncols = 160, 320

images = []
measurements = []
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = prefix + '/IMG/' + filename
    image = cv2.imread(current_path)
    image = image[nrows//2:nrows, :]
    plt.imshow(image)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

raise RuntimeError("...")

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization,Input, Flatten,  Dense
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop



model = Sequential()

# took from https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py
model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(nrows,ncols,3)))

model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))


model.compile(loss = 'mse',  optimizer = 'adam')
model.fit(X_train,  y_train,  validation_split=0.2,  shuffle = True,  nb_epoch=7)


model.save('model.h5')
