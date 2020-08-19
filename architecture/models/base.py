from typing import Callable, Dict, Optional

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os

MODELS_PATH = os.path.dirname(os.path.abspath(__file__))
ARCHITECTURE_PATH = os.path.dirname(MODELS_PATH)
WEIGHTS_PATH = os.path.join(ARCHITECTURE_PATH, 'weights')

class Model:

    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable[..., KerasModel],
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        self.name = '%s_%s_%s' % (self.__class__.__name__, dataset_cls.__name__, network_fn.__name__)

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)

    @property
    def image_shape(self):
        return self.data.input_shape
    
    @property
    def weights_filename(self) -> str:
        os.mkdir(WEIGHTS_PATH, parents=True, exist_ok=True)
        return str(os.path.join(WEIGHTS_PATH, '%s_weights.h5' % self.name))
    
    def fit(
        self, dataset, batch_size: int = 8, epochs: int = 10, callbacks: list = None
    ):
        if callbacks is None:
            callbacks = []
        
        train_augmentation = ImageDataGenerator(rotation_range=15, fill_mode='nearest')

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())
        self.network.fit(
            train_augmentation.flow(dataset.X_train, dataset.y_train, batch_size=batch_size),
            steps_per_epoch=len(dataset.X_train) // batch_size,
            validation_data=(dataset.X_val, dataset.y_val),
            validation_steps=len(dataset.X_val) // batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 8, _verbose: bool = False):
        sequence = ImageDataGenerator(rotation_range=15, fill_mode='nearest').flow(X, y, batch_size=batch_size)
        preds = self.network.predict(sequence)
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))
    
    def loss(self):
        return 'categorical_crossentropy'
    
    def optimizer(self):
        return Adam()
    
    def metrics(self):
        return [keras.metrics.AUC(name='auc'), keras.metrics.CategoricalAccuracy(name='cat_accuracy')]
    
    def load_weights(self):
        self.network.load_weights(self.weights_filename)
    
    def save_weights(self):
        self.network.save_weights(self.weights_filename)
