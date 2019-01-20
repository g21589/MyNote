# -*- coding: utf-8 -*-

import os, sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_data_dir = './data2'

train_dir_1 = './data2/def'
train_dir_2 = './data2/ref'

train_data_gen_dir = './gen'
img_width      = 299
img_height     = 299
batch_size     = 32
classes        = 3

"""
 featurewise_center=False,
 samplewise_center=False,
 featurewise_std_normalization=False,
 samplewise_std_normalization=False,
 zca_whitening=False,
 zca_epsilon=1e-6,
 rotation_range=0.,
 width_shift_range=0.,
 height_shift_range=0.,
 brightness_range=None,
 shear_range=0.,
 zoom_range=0.,
 channel_shift_range=0.,
 fill_mode='nearest',
 cval=0.,
 horizontal_flip=False,
 vertical_flip=False,
 rescale=None,
 preprocessing_function=None,
 data_format=None,
 validation_split=0.0
"""

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

#train_generator = train_datagen.flow_from_directory(
#    directory=train_data_dir,
#    shuffle=False,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    save_to_dir=train_data_gen_dir,
#    save_format='png',
#    save_prefix='',
#    follow_links=False,
#    interpolation='bilinear')

def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height,img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False, 
                                          seed=7)
    
    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height,img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False, 
                                          seed=7)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label
            
            
inputgenerator = generate_generator_multiple(generator=train_datagen,
                                           dir1=train_dir_1,
                                           dir2=train_dir_2,
                                           batch_size=batch_size,
                                           img_height=img_height,
                                           img_width=img_height)

#testgenerator = generate_generator_multiple(test_imgen,
#                                          dir1=train_dir_1,
#                                          dir2=train_dir_2,
#                                          batch_size=batch_size,
#                                          img_height=img_height,
#                                          img_width=img_height)

#if not os.path.exists(train_data_gen_dir):
#    os.makedirs(train_data_gen_dir)
#
#for i in range(1):
#    datas, labels = train_generator.next()
#
#filenames = train_generator.order_filenames
#print("filenames", filenames)
