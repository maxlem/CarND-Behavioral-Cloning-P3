import matplotlib.pyplot as plt
import pickle

with open('history_object.pkl', 'rb') as f:
    history_object = pickle.load(f)
    
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
