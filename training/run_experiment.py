import json
import importlib
from typing import Dict
import os
import tensorflow as tf
import gc
import argparse

import wandb

from training.gpu_manager import GPUManager
from training.util import train_model, plot_confusion_matrix

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_TRAIN_ARGS = {'batch_size': 8, 'epochs': 10}
DEFAULT_OPT_ARGS = {'lr': 1e-3, 'decay': 1e-3 / DEFAULT_TRAIN_ARGS['epochs']}

# experiment_config = {
#     "dataset": "AlzheimerMPRageNoDeep",
#     "dataset_args": {"types": ["CN", "AD"]},
#     "model": "AlzheimerCNN",
#     "network": "mobilenet",
#     "train_args": {'batch_size': 32, 'epochs': 100},
#     "opt_args": {'lr': 1e-4}
#     # "opt_args": {'lr': 1e-4, 'decay': 1e-5} # decay: lr / epochs
# }
# use_wandb = False
# gpu_ind = 0

def run_experiment(experiment_config: Dict, save_weights: bool, gpu_ind: int, use_wandb: bool = True):
    print(f'Running experiment with config {experiment_config}, on GPU {gpu_ind}')

    datasets_module = importlib.import_module('architecture.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    dataset_args = experiment_config.get('dataset_args', {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)

    models_module = importlib.import_module('architecture.models')
    model_class_ = getattr(models_module, experiment_config['model'])

    networks_module = importlib.import_module('architecture.networks')
    network_fn_ = getattr(networks_module, experiment_config['network'])
    network_args = experiment_config.get('network_args', {})

    opt_args_ = experiment_config.get('opt_args', DEFAULT_OPT_ARGS)

    model = model_class_(
        dataset_cls=dataset_class_, network_fn=network_fn_, dataset_args=dataset_args, network_args=network_args, opt_args=opt_args_
    )

    experiment_config["train_args"] = {
        **DEFAULT_TRAIN_ARGS,
        **experiment_config.get("train_args", {}),
    }

    experiment_config["opt_args"] = {
        **DEFAULT_OPT_ARGS,
        **experiment_config.get("opt_args", {}),
    }

    experiment_config["experiment_group"] = experiment_config.get("experiment_group", None)
    experiment_config["gpu_ind"] = gpu_ind

    if use_wandb:
        dataset_name = {
            'AlzheimerT2SmallDataset': 't2mini',
            'AlzheimerT2StarSmallDataset': 't2starmini',
            'AlzheimerT2StarFullDataset': 't2starfull',
            'AlzheimerMPRageDeep': 'mprage_deep',
            'AlzheimerMPRageNoDeep': 'mprage_nodeep',
        }
        tags = []
        tags.append('-'.join(list(dataset.mapping.values())).lower())

        if dataset.num_classes > 2:
            tags.append('binary')
        else:
            tags.append('multiclass')

        wandb.init(
            project='alzheimer-dl',
            config=experiment_config,
            name='{model} {dataset} {epochs}ep {batch_size}bs'.format(
                model=('multi ' if dataset.num_classes > 2 else 'bin ') + experiment_config['network'],
                dataset=dataset_name[experiment_config['dataset']],
                epochs=experiment_config["train_args"]["epochs"],
                batch_size=experiment_config["train_args"]["batch_size"],
                tags=tags
            )
        )

    with tf.device('/GPU:0'):
        train_model(
                model,
                dataset,
                epochs=experiment_config["train_args"]["epochs"],
                batch_size=experiment_config["train_args"]["batch_size"],
                use_wandb=use_wandb,
        )

    if use_wandb:
        classes = list(dataset.mapping.values())

        cm_val = plot_confusion_matrix(model, dataset.X_val, dataset.y_val, classes, experiment_config["train_args"]["batch_size"])
        wandb.log({"confusion_matrix - validation data": cm_val})

        cm_test = plot_confusion_matrix(model, dataset.X_test, dataset.y_test, classes, experiment_config["train_args"]["batch_size"])
        wandb.log({"confusion_matrix - test": cm_test})

    if save_weights:
        model.save_weights()


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="Provide index of GPU to use.")
    parser.add_argument(
        "--save",
        default=False,
        dest="save",
        action="store_true",
        help="If true, then final weights will be saved to canonical, version-controlled location",
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experimenet JSON (\'{"dataset": "AlzheimerT2SmallDataset", "model": "MultiClassCNN", "network": "vgg16"}\'',
    )
    parser.add_argument(
        "--nowandb", default=False, action="store_true", help="If true, do not use wandb for this run",
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()

    if args.gpu < 0:
        gpu_manager = GPUManager()
        args.gpu = gpu_manager.get_free_gpu()  # Blocks until one is available

    experiment_config = json.loads(args.experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_experiment(experiment_config, args.save, args.gpu, use_wandb=not args.nowandb)


if __name__ == "__main__":
    main()
    gc.collect()
