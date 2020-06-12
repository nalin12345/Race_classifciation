#imports
import tensorflow as tf
import os
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

#re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = r'C:\Users\Nalin\Downloads\Racist\Train'
valid_path = r'C:\Users\Nalin\Downloads\Racist\Train'

#add preprocessing layer to the front of VGG
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#dont train existing weights
for layer in vgg.layers:
    layer.trainable=False

#useful for getting number of classes
folders = glob(r'C:\Users\Nalin\Downloads\Racist\Train\*')

# our layers - you can add more if yu want
x = Flatten()(vgg.output)
# x =Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

#create a model object
model = Model(inputs=vgg.input, outputs=prediction)

#view the structure of model
model.summary()

#tell the model what cost and optimisation method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=.15,
                                   vertical_flip=True,
                                   featurewise_center=True,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   zca_whitening=False,
                                   zca_epsilon=1e-6,                                  
                                   shear_range=0.1,
                                   height_shift_range=.15,
                                   rotation_range=0,
                                   horizontal_flip=True,
                                   brightness_range=None,
                                   channel_shift_range=0.,
                                   fill_mode='nearest',
                                   cval=0.,
                                   preprocessing_function=None,
                                   data_format='channels_last',
                                   validation_split=0.0
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\Nalin\Downloads\Racist\Train',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'C:\Users\Nalin\Downloads\Racist\Test',
                                            target_size=(224, 224),
                                            batch_size= 32,
                                            class_mode='categorical')
                                                
'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model
r=model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

#loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('new_model.h5')

