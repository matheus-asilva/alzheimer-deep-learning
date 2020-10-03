from typing import Tuple
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def mobilenet(
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, ...],
    weights: str = 'imagenet',
    include_top: bool = False
) -> Model:

    base_model = MobileNet(
        weights=weights,
        include_top=include_top,
        input_tensor=Input(shape=input_shape)
    )

    if include_top:
        return base_model
    else:
        # Construct the head of the model that will be placed on top of the base model
        num_classes = output_shape[0]
        head_model = base_model.output
        head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
        head_model = Flatten(name='flatten')(head_model)
        head_model = Dense(64, activation='relu')(head_model)
        head_model = Dropout(0.7)(head_model)

        if num_classes > 2:
            activation = 'softmax'
        else:
            activation = 'sigmoid'

        head_model = Dense(num_classes, activation=activation)(head_model)

        # Place the head Fully Connected model on top of the base model (actual model)
        model = Model(base_model.input, head_model)

        # Loop over all layers in the base model and freeze them so they won't be updated during the first training process
        for layer in base_model.layers:
            layer.trainable = False

        return model
