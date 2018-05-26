import csv
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import math
import random
from scipy.stats import norm
import time


################## SECTION 1 : Dataset parsing and analysis ####################

prefixes = ['./ds{0}'.format(i) for i in [0,6,7,8]] # all my dataset folders are prefixed with 'ds' and suffixed with a number

lines = []
line_directories = [] #each line can come from a different directory, prefixes remembers it
for prefix in prefixes:
    with open(prefix + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            line_directories.append(prefix)

samples = []
angles = []

indices = np.arange(len(lines))

print("Number of csv lines: {:}".format(len(lines)))

np.random.shuffle(indices) #we will read lines in random order

for i in indices:
    
    line = lines[i]
    def add_image(source_path, angle):
        filename = source_path.split('/')[-1]
        angles.append(angle)
        samples.append([line_directories[i] + '/IMG/' + filename, angle])
    
    angle = float(line[3])
    
    add_image(line[0], angle) # center
    
    if abs(angle) > 0.15 : #helps reducing bias for small angles
        add_image(line[1], angle + .2) # left
        add_image(line[2], angle -.2) # right


# dataset distribution analysis
hist, bins = np.histogram(angles, bins = 15)

def angle_hist_i(angle, bins):
    '''
        return bin index for a give angle
    '''
    return np.searchsorted(bins, [angle]) - 1 

arg_angle_zero = angle_hist_i(0, bins) #we ignore angle 0 since it dominates all dataset
hist_mask = np.arange(len(hist)) != arg_angle_zero
hist_mean = np.mean(hist[hist_mask])
hist_std = np.std(hist[hist_mask]) 

# original distribution:
plt.bar(bins[:-1], hist, bins[1] - bins[0])
plt.show()

target_hist = np.clip(hist, hist_mean - hist_std/2, hist_mean + hist_std/2)

#target distribution:
plt.bar(bins[:-1], target_hist, bins[1] - bins[0])
plt.show() 

# masks used in the dataset generator
is_bin_scarse = hist < hist_mean
is_bin_very_scarse = hist < (hist_mean - hist_std/2)

import sklearn
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


############################## SECTION 2 : Model definition and training ################################3

def random_changes(yuv_img, scale = 64):
    ''' 
    modify randomly  YUV image brightness, and a random
    vertical shift of the horizon position
    '''
    # random brightness
    float_img = yuv_img.astype(float)
    float_img[:,:,0] += np.random.randint(-scale, scale)
    
    
    # random circle shadow
    h,w,_ = float_img.shape
    
    circle = np.zeros((h,w))
    
    cv2.circle(circle, center = (np.random.randint(h/4,3*h/4), np.random.randint(w/4,3*w/4))
               , radius = np.random.randint(16, 128)
               , color = (1,1,1)
               , thickness = -1) #fills circle
    float_img[:,:,0] -=  (np.random.rand() * 0.25 + 0.25) * circle * 255 #random is scaled between .25 and .5
    
    # return and clip values between (0,255)
    return np.clip(float_img, 0, 255).astype(np.uint8)

def generator(samples, batch_size=32, validation = False):
    num_samples = len(samples)
    np_angles = np.array(samples)[:,1].astype(np.float32)
    target_batch_hist = np.ceil(target_hist * batch_size / num_samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            start = time.time()
            batch_samples = samples[offset:offset+batch_size]
            
             #shuffle each batch each time, since we won't keep all samples in the batch, 
             #as we want the optimizer to see different part of the skipped data every epoch:
            random.shuffle(batch_samples)
            
            batch_hist = np.zeros_like(hist)
            h = np.zeros_like(hist)
            images = []
            angles = []
            
            
            def add_sample(image, angle, hist_i):
                
                if len(images) >= batch_size: #in fact it can only be <=
                    return 0
                
                if validation or is_bin_scarse[hist_i] or batch_hist[hist_i] < target_batch_hist[hist_i]:
                    images.append(image)
                    angles.append(angle)
                    batch_hist[hist_i] += 1
                    return 1
                    
                return 0
                
            while len(images) < batch_size : 
                n_new = 0
                for (path, angle) in batch_samples:
                    
                    hist_i = angle_hist_i(angle, bins)
                    h[hist_i] += 1
                    
                    # yuv allows the nn to focus on intensity and color information separately, was recommended in the nVidia paper
                    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2YUV) #do NOT forget to mod drive.py
                    n_new += add_sample(image, angle, hist_i)
                    
                    if validation :
                        continue
                    
                    #data augmentation (only for scarse data): 
  
                    if is_bin_scarse[hist_i] : 
                        #flip image and steering angle
                        rev_angle = angles * -1
                        rev_hist_i = angle_hist_i(rev_angle, bins)
                        flipped = cv2.flip(image,1)
                        n_new += add_sample(flipped, rev_angle, rev_hist_i)
                            
                        if is_bin_very_scarse[hist_i]:
                            #further data augmentation for very rare data
                            n_new += add_sample(random_changes(flipped), rev_angle, rev_hist_i)
                            n_new += add_sample(random_changes(image), angle, hist_i)
                    
                    if len(images) >= batch_size: #in fact it can only be <=
                        break;
                
                if n_new == 0 or validation:
                    break;
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            

from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Reshape
from keras.layers import Flatten,  Dense, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

def nvidia_model(nrows = 160, ncols = 320):

    model = Sequential()

    # common weights regularizer, to reduce overfitting:
    regularizer = l2(0.001)
    
    #crop the image to reduce weight and improve NN's focus
    model.add(Cropping2D(cropping=((75, 20), (0, 0)), input_shape=(nrows,ncols,3)))
    
    #image normalization, not sur it is needed on yuv images, but It can only help
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    #convolution pipeline, subsampling and relu activations
    model.add(Convolution2D(16,5,5,border_mode='valid', activation='relu', subsample=(2,2), W_regularizer = regularizer))
    model.add(Convolution2D(32,5,5,border_mode='valid', activation='relu', subsample=(2,2), W_regularizer = regularizer))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1), W_regularizer = regularizer))
    
    #dense pipeline
    model.add(Flatten())
    model.add(Dense(100, activation='relu', W_regularizer = regularizer))
    model.add(Dense(50, activation='relu', W_regularizer = regularizer))
    model.add(Dense(25, activation='relu', W_regularizer = regularizer))
    model.add(Dense(10, activation='relu', W_regularizer = regularizer))

    #regression output, tanh so that output can be positive or negative
    model.add(Dense(1, activation='tanh'))
    
    return model

model = nvidia_model() 

print(model.summary())
# resume training...
#from keras.models import load_model
#del model
#model = load_model("model.h5")

nb_epoch = 10
batch_size = 128

train_generator = generator(train_samples, batch_size=batch_size, validation = False)
validation_generator = generator(validation_samples, batch_size=batch_size, validation = True)

model.compile(loss = 'mse',  optimizer = 'adam')

checkpointer = ModelCheckpoint(filepath="model_ckpt.h5", verbose=1, save_best_only=True)

history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=nb_epoch, callbacks=[checkpointer])

model.save('model.h5')

# list all data in history credit: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

