from typing import Callable, Dict, Tuple

import numpy as np

from architecture.models.base import Model
from architecture.networks import vgg16
from architecture.datasets.alzheimert2_small_dataset import AlzheimerT2SmallDataset


class AlzheimerCNN(Model):

    def __init__(
        self,
        dataset_cls: type = AlzheimerT2SmallDataset,
        network_fn: Callable = vgg16,
        dataset_args: Dict = None,
        network_args: Dict = None,
        opt_args: Dict = None):

        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args, opt_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)

        predict_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(predict_raw)
        confidence_of_prediction = predict_raw[ind]
        predicted_class = self.data.mapping[ind]
        return predicted_class, confidence_of_prediction
