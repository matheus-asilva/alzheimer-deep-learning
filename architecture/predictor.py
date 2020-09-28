from typing import Tuple, Union
import numpy as np

from architecture.models import MultiClassCNN
import architecture.util as util

class ImagePredictor:

    def __init__(self):
        """Given an image of an exam, recognizes it"""
        self.model = MultiClassCNN
        self.model.load_weights()
    
    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on single image"""
        if isinstance(image_or_filename, str):
            image = util.load_image(image_or_filename)
        else:
            image = image_or_filename
        return self.model.predict_on_image(image)
    
    def evaluate(self, dataset):
        """Evaluate on a dataset"""
        return self.model.evaluate(dataset.X_val, dataset.y_val)