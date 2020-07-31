import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import History
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 150


# Read in data:
data = np.load('data.npy')
data = data/100.
#data = data.reshape(2000, 512, 16)
plt.imshow(data[4], cmap='gray_r', aspect='auto')
plt.show()
plt.imshow(data[5], cmap='gray_r', aspect='auto')
plt.show()

labels = np.load('labels.npy')


# Split the data
x, x_test, y, y_test = train_test_split(data, labels, test_size=0.1, shuffle= True)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle= True)


# Setup CNN:
adam = Adam(lr=0.00005)
history = History()
nnet = Sequential()

nnet.add(Conv2D (16, kernel_size=(3,3), input_shape=(512,16,3), activation='relu'))
nnet.add(MaxPooling2D (pool_size=(2,2)))
nnet.add(Dropout (0.2))

nnet.add(Conv2D (32, (3,3), activation='relu'))
nnet.add(MaxPooling2D (pool_size=(2,2)))
nnet.add(Dropout (0.2))

nnet.add(Flatten())
nnet.add(Dense (32, activation='relu'))
nnet.add(Dropout (0.25))

nnet.add(Dense (8, activation='relu'))
#nnet.add(Dropout (0.5))
nnet.add(Dense (1, activation='sigmoid'))

nnet.compile (loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

history = nnet.fit(x_train, y_train, shuffle=True, batch_size=50, epochs=10, 
                   verbose=1, validation_split=0.0, validation_data=(x_valid, y_valid))


# Print test accuracy:
test_loss, test_acc = nnet.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


# Plot training & validation loss and accuracy values:
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()