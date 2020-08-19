"""AlexNet network."""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential

def alexnet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:

    num_classes = output_shape[0]

    # First block
    model = Sequential()
    model.add(Conv2D(96, (3,3), strides=(4,4), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, strides=(2,2)))

    # Second block
    model.add(Conv2D(256, (11,11), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Third block
    model.add(Conv2D(384, (3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Fourth block
    model.add(Conv2D(384, (3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Fifth block
    model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, strides=(2,2)))

    # Flatten block
    model.add(Flatten())

    # Fully connected block 1
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Fully connected block 2
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model