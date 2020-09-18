from time import time

from tensorflow.keras.callbacks import EarlyStopping, Callback

import wandb
from wandb.keras import WandbCallback

from architecture.datasets.dataset import Dataset
from architecture.models.base import Model

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import numpy as np

EARLY_STOPPING = True

class WandbImageLogger(Callback):
    """Custom callback for logging image predictions"""

    def __init__(self, model_wrapper: Model, dataset: Dataset, example_count: int = 10):
        super().__init__()
        self.model_wrapper = model_wrapper

        example_count = np.random.choice(len(dataset.X_val), example_count).tolist()
        self.val_images = dataset.X_val[example_count]

    def on_epoch_end(self, epoch, logs=None):
        images = [
            wandb.Image(image, caption='{}: {}'.format(*self.model_wrapper.predict_on_image(image)))
            for i, image in enumerate(self.val_images)
        ]
        wandb.log({'examples': images}, commit=False)
    

def train_model(model: Model, dataset: Dataset, epochs: int = 10, batch_size: int = 8, use_wandb: bool = False) -> Model:
    """Train model"""
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_auc', min_delta=0, patience=10, verbose=1, mode='max', restore_best_weights=True)
        callbacks.append(early_stopping)
    
    if use_wandb:
        image_callback = WandbImageLogger(model, dataset)
        wandb_callback = WandbCallback()
        callbacks.append(image_callback)
        callbacks.append(wandb_callback)
    
    model.network.summary()

    t = time()
    _history = model.fit(dataset=dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    print('Training took {:2f} minutes'.format((time() - t) / 60))

    return model

def plot_confusion_matrix(model, X, y, classes, batch_size=8, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    if len(y[0]) > 1:
        y = np.argmax(y, axis=1)
    
    y_preds = model.predict(X=X, batch_size=batch_size)

    cm = confusion_matrix(y_true=y, y_pred=y_preds)
    
    g = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return g