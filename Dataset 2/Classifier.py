#Classifier for dataset 2
import numpy as np
from data_import import import_data
from shuffle_in_unison import shuffle_in_unison
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

#Import data
labels, images, dataset=import_data('Cancer labels.csv', 'cancer_images')
images=np.reshape(images, [20,400,640,1])

labels=to_categorical(labels)

#Create model
#Based on VGG-like net
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(400, 640, 1)))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optimizer = adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

print(model.summary())

#Train on augmented data set

batch_size=1
validation_split=0.2

#Shuffle the data, since the data is currently sorted by label, so the validation part will get 100% one data type
images, labels=shuffle_in_unison(images,labels)
images=np.float16(images/255.0)

#Augments the dataset, as it is quite small
datagen=ImageDataGenerator(
        rotation_range=360,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split)# I don't think this helped

history=model.fit_generator(datagen.flow(images, labels, batch_size=batch_size, subset='training'),
                            epochs=100, steps_per_epoch=images.shape[0]*(1-validation_split)//batch_size,
                            validation_data=datagen.flow(images, labels, batch_size=batch_size, subset='validation'),
                            validation_steps=dataset.shape[0]*validation_split//batch_size)
                           
#history = model.fit(images, labels, batch_size=1, epochs=100, validation_split=0.25)

model.save('saved_weights.h5')