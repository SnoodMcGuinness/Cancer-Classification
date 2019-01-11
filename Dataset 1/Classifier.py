import numpy as np
from data_import import import_data
import pandas as pd
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model
#Compile options
optimizer='rmsprop'
loss='binary_crossentropy'
metrics=['accuracy']

#Import data
dataset, labels, features = import_data('Exercise1 - data.csv')

#Create the model, based on the 'MLP for binary classification' from https://keras.io/getting-started/sequential-model-guide/
model = Sequential()

model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print(model.summary())
#plot_model(model,to_file='model.png') # This will not run on my PC since I have installed plaidml so I could use my AMD GPU, which requires quite specific versions of everything, and I could not get pydot to work with it

#Fit the model
history = model.fit(features, labels, epochs=500, batch_size=100, validation_split=0.2)

#Plot the training history
epochs=np.arange(len(history.history['val_loss']))
plt.plot(epochs, history.history['loss'], label='loss')
plt.plot(epochs, history.history['acc'], label='accuracy')
plt.plot(epochs, history.history['val_loss'], label='val_loss')
plt.plot(epochs, history.history['val_acc'], label='val_accuracy')
plt.legend(loc='upper left')
plt.show()