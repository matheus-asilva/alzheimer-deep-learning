import os
from tqdm import tqdm
import shutil
from itertools import chain
from utils import get_unique_ids_and_paths, sampling


def save_images(ids, images, disease, sample, output_path):

    # Cria lista com caminhos dos ids selecionados
    paths = []
    for id in ids[sample]:
        paths.append([image for image in images if id in image])

    # Desencadeia listas
    paths = list(chain(*paths))

    # Testa condições
    if sample not in ['train', 'validation', 'test']:
        raise "Must be ['train', 'validation', 'test']"

    if disease not in ['AD', 'CN', 'MCI']:
        raise "Must be ['AD', 'CN', 'MCI']"

    # Verifica se caminho existe
    root = os.path.join(output_path, sample, disease)
    if not os.path.exists(root):
        os.makedirs(root)

    # Copia imagens para pasta de destino
    for image in tqdm(paths):
        shutil.copy2(image, root)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input_path", help="Input path with main disease folder", required=True)
    parser.add_argument("--output_path", help="Output path to send sampled images", required=True)
    parser.add_argument("--sample", help="Type of sample", choices=["train", "validation", "test"], required=True)
    args = parser.parse_args()

    disease = args.input_path.split(os.path.sep)[-1]
    patient_ids, images = get_unique_ids_and_paths(args.input_path)
    ids = sampling(patient_ids)
    save_images(ids, images, disease, args.sample, args.output_path)
