from typing import Tuple
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model


def vgg19(input_shape: Tuple[int, int, int], output_shape: Tuple[int, ...], weights: str='imagenet', include_top: bool=False) -> Model:

    base_model = VGG19(weights=weights, include_top=include_top, input_tensor=Input(shape=input_shape))
    if include_top:
        return base_model
    else:
        # Construct the head of the model that will be placed on top of the base model
        num_classes = output_shape[0]
        head_model=base_model.output
        head_model=AveragePooling2D(pool_size=(4, 4))(head_model)
        head_model=Flatten(name='flatten')(head_model)
        head_model=Dense(64, activation='relu')(head_model)
        head_model=Dropout(0.5)(head_model)
        head_model=Dense(num_classes, activation='softmax')(head_model)

        # Place the head Fully Connected model on top of the base model (actual model)
        model=Model(base_model.input, head_model)

        # Loop over all layers in the base model and freeze them so they won't be updated during the first training process
        for layer in base_model.layers:
            layer.trainable=False

        return model
