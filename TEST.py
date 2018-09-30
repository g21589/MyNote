# -*- coding: utf-8 -*-

import os, sys

import numpy as np
import pandas as pd

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

train_data_dir = './data2'
train_dir_1 = './data2/def'
train_dir_2 = './data2/ref'

train_data_gen_dir  = './gen'
train_data_gen_dir1 = './gen1'
train_data_gen_dir2 = './gen2'

img_width      = 299
img_height     = 299
batch_size     = 4
classes        = 3

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

mkdirs(train_data_gen_dir1)
mkdirs(train_data_gen_dir2)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    zca_whitening=True,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

#train_generator = train_datagen.flow_from_directory(
#    train_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size)

def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height,img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=True,
                                          seed=7,
                                          save_to_dir=train_data_gen_dir1,
                                          save_format='png',
                                          save_prefix='gen')
    
    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height,img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=True, 
                                          seed=7,
                                          save_to_dir=train_data_gen_dir2,
                                          save_format='png',
                                          save_prefix='gen')
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
#        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label
        X = np.concatenate((X1i[0], X2i[0]), axis=3)
        #print(X1i[0].shape)
        #print(X.shape)
        yield X, X2i[1]

train_generator = generate_generator_multiple(generator=train_datagen,
                                           dir1=train_dir_1,
                                           dir2=train_dir_2,
                                           batch_size=batch_size,
                                           img_height=img_height,
                                           img_width=img_height)

# create the base pre-trained model
base_model = InceptionV3(weights=None, 
                         include_top=False,
                         input_shape=(img_width, img_height, 6))
#base_model = InceptionResNetV2(weights='./pretrain_models/Inception_Resnet_v2_notop.h5', include_top=False)
#base_model = NASNetLarge(weights=None, include_top=False)
#base_model.load_weights('./pretrain_models/NASNet_large_notop.h5', by_name=True)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# train the model on the new data for a few epochs
model.fit_generator(
        train_generator, 
        steps_per_epoch=4, 
        epochs=2,
        use_multiprocessing=False)

#model.fit_generator(
#    train_generator,
#    steps_per_epoch=32,
#    epochs=10,
#    validation_data=validation_generator,
#    validation_steps=nb_validation_samples)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

## let's visualize layer names and layer indices to see how many layers
## we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)
#
## we chose to train the top 2 inception blocks, i.e. we will freeze
## the first 249 layers and unfreeze the rest:
#for layer in model.layers[:249]:
#   layer.trainable = False
#for layer in model.layers[249:]:
#   layer.trainable = True
#
## we need to recompile the model for these modifications to take effect
## we use SGD with a low learning rate
#from keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
#
## we train our model again (this time fine-tuning the top 2 inception blocks
## alongside the top Dense layers
#model.fit_generator(...)

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
