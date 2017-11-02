# -*- coding: utf-8 -*-

"""
    this file will use trained model to evaluate test data
"""

import os
import sys

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

test_dir = './datasets/test/'
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(50, 50),
    batch_size=20,
    class_mode='categorical')

assert len(sys.argv) >= 2

model_name = sys.argv[1]

model = load_model(os.path.join(os.getcwd(), model_name))

loss, acc = model.evaluate_generator(test_generator, 20)

print('loss: {} \n acc: {}'.format(loss, acc))




