# -*- coding: utf-8 -*-

import os
import sys
import shutil
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

models_save_dir = './models/'

if not os.path.exists(models_save_dir):
    os.mkdir(models_save_dir)

dataset_dir = './datasets/raw_datasets/Images/'

train_dir = './datasets/train/'

validation_dir = './datasets/validation/'

test_dir = './datasets/test/'

# if the second arguments is '-n' then split data again
if len(sys.argv) >= 2 and sys.argv[1] == '-n':

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.mkdir(train_dir)

    os.mkdir(validation_dir)

    os.mkdir(test_dir)

    for i in range(0, 43):
        #一共43个分类，每个循环一次，按照8:1:1的比例分配 训练/validation/测试 数据
        category = i
        foldername = str(i).zfill(5)
        foldername_new = str(i)


        dataset_path = os.path.join(dataset_dir, foldername)
        train_path = os.path.join(train_dir, foldername_new)
        os.mkdir(train_path)
        validation_path = os.path.join(validation_dir, foldername_new)
        os.mkdir(validation_path)
        test_path = os.path.join(test_dir, foldername_new)
        os.mkdir(test_path)

        dataset = np.array(os.listdir(dataset_path))
        np.random.shuffle(dataset)

        #train_dataset, test_dataset = train_test_split(dataset, target, test_size=0.2)
        """

        train_test_split method raise 'too many values to unpack' error 
        so use array slice simplely

        """

        train_dataset = dataset[0:int(len(dataset)*0.8)]
        validation_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
        test_dataset = dataset[int(len(dataset)*0.9):]

        for train_item in train_dataset:
            im = Image.open(os.path.join(dataset_path, train_item))
            im.save(os.path.join(train_path, train_item.split('.')[0] + '.png'))
            #shutil.copy(os.path.join(dataset_path, train_item), train_path)

        for validation_item in validation_dataset:
            im = Image.open(os.path.join(dataset_path, validation_item))
            im.save(os.path.join(validation_path, validation_item.split('.')[0] + '.png'))
            #shutil.copy(os.path.join(dataset_path, validation_item), validation_path)

        for test_item in test_dataset:
            im = Image.open(os.path.join(dataset_path, test_item))
            im.save(os.path.join(test_path, test_item.split('.')[0] + '.png'))
            #shutil.copy(os.path.join(dataset_path, test_item), test_path)

"""
    clear_session every trian
"""

K.clear_session()


batch_size = 10  
steps_per_epoch = int(sum([len(files) for r, d, files in os.walk(train_dir)])/batch_size)

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(90, 90, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(43, activation='softmax'))

"""
    check our model summary
"""
#model.summary()

model.compile(loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['accuracy']
)

"""
    start processing input data
    turn raw image to numpy array
"""

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(90,90),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(90,90),
    batch_size=batch_size,
    class_mode='categorical')

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

history = model.fit_generator(train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=15,
    callbacks=[earlystopping])

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(90,90),
    batch_size=20,
    class_mode='categorical')

loss, acc = model.evaluate_generator(test_generator, 20)

model.save(os.path.join(models_save_dir, 'traffic_' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S') +  '_' + str(acc) + '.h5'))







