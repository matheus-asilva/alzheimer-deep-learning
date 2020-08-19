"""VGG16 architecture"""
from typing import Tuple
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import get_source_inputs
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Input
from tensorflow.python.keras.engine import training

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

def vgg16(input_shape: Tuple[int, int, int], output_shape: Tuple[int, ...], input_tensor = None, weights: str = 'imagenet', include_top: bool = False) -> Model:

    if weights == 'imagenet':
        if include_top:
            weights_path = get_file(
                    'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                    WEIGHTS_PATH,
                    cache_subdir='models',
                    file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
    elif weights is None:
        weights_path = weights

    # First block
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1', input_shape=(input_shape)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), name='block1_pool'))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), name='block2_pool'))

    model.add(Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), name='block3_pool'))

    model.add(Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), name='block4_pool'))

    model.add(Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), name='block5_pool'))
    
    if include_top:
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dense(1000, activation='softmax', name='predictions'))
        
        final_model = Model(model.input, model.output)
        final_model.load_weights(weights_path)

    else:
        model.load_weights(weights_path)
        
        num_classes = output_shape[0]
        head_model = model.output
        head_model = AveragePooling2D(pool_size=(4,4), name='pool')(head_model)
        head_model = Flatten(name='flatten')(head_model)
        head_model = Dense(64, activation='relu')(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(num_classes, activation='softmax', name='predictions')(head_model)
        
        final_model = Model(model.input, head_model)

        for layer in model.layers:
            layer.trainable = False
    
    return final_model