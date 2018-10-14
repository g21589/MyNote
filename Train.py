# -*- coding: utf-8 -*-

import os, sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import confusion_matrix_pretty_print

#%% Configs
train_data_dir = os.path.normpath('./Datasets/flower_split_0.30_1234/Train')
valid_data_dir = os.path.normpath('./Datasets/flower_split_0.30_1234/Valid')
checkpoint_fn  = os.path.normpath('./Models/Model.{epoch:02d}-{val_acc:.2f}.h5')
#train_dir_1 = os.path.normpath('./data2/def')
#train_dir_2 = os.path.normpath('./data2/ref')

#train_data_gen_dir  = os.path.normpath('./gen')
#train_data_gen_dir1 = os.path.normpath('./gen1')
#train_data_gen_dir2 = os.path.normpath('./gen2')

img_size       = 299
channels       = 3
classes        = 5
batch_size     = 32

#%%
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

#mkdirs(train_data_gen_dir1)
#mkdirs(train_data_gen_dir2)
if not os.path.isdir(os.path.dirname(checkpoint_fn)):
    mkdirs(os.path.dirname(checkpoint_fn))

#%% ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size
)

valid_datagen = ImageDataGenerator(
    rescale=1.0/255
)

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False
)

#def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
#    genX1 = generator.flow_from_directory(dir1,
#                                          target_size=(img_height,img_width),
#                                          class_mode='categorical',
#                                          batch_size=batch_size,
#                                          shuffle=True,
#                                          seed=7,
#                                          #save_to_dir=train_data_gen_dir1,
#                                          #save_format='png',
#                                          #save_prefix='gen'
#                                          )
#    
#    genX2 = generator.flow_from_directory(dir2,
#                                          target_size=(img_height,img_width),
#                                          class_mode='categorical',
#                                          batch_size=batch_size,
#                                          shuffle=True, 
#                                          seed=7,
#                                          #save_to_dir=train_data_gen_dir2,
#                                          #save_format='png',
#                                          #save_prefix='gen'
#                                          )
#    while True:
#        X1i = genX1.next()
#        X2i = genX2.next()
#        X = np.concatenate((X1i[0], X2i[0]), axis=3)
#        yield X, X2i[1]
#
#train_generator = generate_generator_multiple(generator=train_datagen,
#                                           dir1=train_dir_1,
#                                           dir2=train_dir_2,
#                                           batch_size=batch_size,
#                                           img_height=img_size,
#                                           img_width=img_size)

#%% Define network
# create the base pre-trained model
base_model = InceptionV3(weights=None, 
                         include_top=False,
                         input_shape=(img_size, img_size, channels))
#base_model = InceptionResNetV2(weights=None, 
#                         include_top=False,
#                         input_shape=(img_size, img_size, channels))
#base_model = NASNetLarge(weights=None, 
#                         include_top=False,
#                         input_shape=(img_size, img_size, channels))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
#x = Dropout(0.2)(x)
predictions = Dense(classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#for layer in base_model.layers:
#    layer.trainable = True

model.compile(
    optimizer='adam', # adam, rmsprop
    loss='categorical_crossentropy', 
    metrics=[
        'acc', 
#        precision, 
#        recall, 
#        fmeasure
    ]
)

#model.summary()

shapes_count = int(np.sum([np.prod(np.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))
g_memory = shapes_count * 4 / 1024 / 1024 / 1024
print('Model memory size: %.4f (GB)' % (g_memory))

#%% Define callbacks
cb_EarlyStop  = EarlyStopping(monitor='val_acc', min_delta=0.05, patience=10, verbose=2, mode='auto', baseline=None)
cb_Checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_ReduceLR   = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=2, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

#%% Training
# train the model on the new data for a few epochs
history = model.fit_generator(
    generator           = train_generator,
    validation_data     = valid_generator,
    steps_per_epoch     = len(train_generator), 
    epochs              = 3,
    max_queue_size      = 16,
    workers             = 4,
    use_multiprocessing = False,
    callbacks=[
        cb_EarlyStop,
        cb_Checkpoint, 
        cb_ReduceLR
    ]
)

model.save_weights('Model_NASNetL.h5')

#%% Predict vaildation set
valid_generator.reset()
pred_probs = model.predict_generator(valid_generator)

class2index = valid_generator.class_indices
index2class = dict((v,k) for k,v in class2index.items())
labels      = list(index2class.values())

true_index = valid_generator.classes
true_label = list(map(lambda x: index2class[x], true_index))

pred_index = np.argmax(pred_probs, axis=1)
pred_label = list(map(lambda x: index2class[x], pred_index))

conf_mat = confusion_matrix(true_label, pred_label, labels=labels)

cm_df = pd.DataFrame(conf_mat, index=labels, columns=labels)
confusion_matrix_pretty_print.pretty_plot_confusion_matrix(
        cm_df, cmap='PuRd', cbar=True, show_null_values=2, fz=9, figsize=[6,5])

#%% Plot result
fig = plt.figure(figsize=(8, 3))

ax1 = plt.subplot(1, 2, 1)
ax1.plot(history.history['acc'], marker='.')
ax1.plot(history.history['val_acc'], marker='.')
ax1.set_title('Model accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.grid(color='#CCCCCC00')
ax1.legend(['Train', 'Test'], loc='upper left')

ax2 = plt.subplot(1, 2, 2)
ax2.plot(history.history['loss'], marker='.')
ax2.plot(history.history['val_loss'], marker='.')
ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.grid(color='#CCCCCC00')
ax2.legend(['Train', 'Test'], loc='upper right')

plt.show()
