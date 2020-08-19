import os
import re
import numpy as np
from tqdm import tqdm
import shutil
import random
from itertools import chain
import argparse

def organiza_imagens(path):
  filepath = []
  ids = []
  for root, dirs, files in os.walk(path):
    for f in tqdm(files):
      if f.endswith('.png') and ('AXIAL' in f.upper()) and ('FLAIR' not in f.upper()):
        if f.split(' ')[-1][:-4] in [f'SLICE_{str(i)}' for i in np.arange(22, 33)]:
          
          # Mapeio caminho do png
          fullpath = os.path.join(root, f)
          filepath.append(fullpath)
          
          # Mapeio id do paciente
          dir = root.split(os.path.sep)[-1]
          ids.append(re.search(f'(?<={dir}\\\\)(.*)(?= AXIAL)', fullpath.upper()).group(1))
  return list(np.unique(ids)), filepath

def reamostragem(lista):
  random.seed(42)
  
  random.shuffle(lista)

  train_index = int(np.round(len(lista) * .7))
  val_index = int(np.round(len(lista) * .2))
  test_index = int(np.round(len(lista) * .1))

  return {
      'train': lista[:train_index], 
      'validation': lista[train_index: train_index + val_index], 
      'test': lista[-test_index:]
      }

def salva_imagens(lista_ids, lista_imagens, tipo, amostra):
  # Cria lista com caminhos dos ids selecionados
  paths = []
  for id in lista_ids[amostra]:
    paths.append([imagem for imagem in lista_imagens if id in imagem])
  
  # Desencadeia listas
  paths = list(chain(*paths))

  # Verifica se caminho existe
  root = os.path.join(DATA_DIR, 'filtered', amostra, tipo)
  if root not in os.listdir(DATA_DIR):
    os.makedirs(root)

  # Copia imagens para pasta de destino
  for imagem in tqdm(paths):
    shutil.copy2(imagem, os.path.dirname(imagem.replace('raw', f'filtered\\{amostra}')))

if __name__ == '__main__':
    
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--tipo', required=True, choices=['AD', 'CN', 'MCI'], help='Classe a ser predita')
    parser.add_argument('--amostra', required=True, choices=['train', 'validation', 'test'], help='Amostra')
    args = parser.parse_args()

    print('Lendo imagens...')
    id_paciente, imagens = organiza_imagens(os.path.join(DATA_DIR, 'raw', args.tipo))
    
    print('Separando amostras...')
    ids = reamostragem(id_paciente)

    print('Salvando imagens...')
    salva_imagens(ids, imagens, args.tipo, args.amostra)