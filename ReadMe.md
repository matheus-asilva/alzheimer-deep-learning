# Classifying Alzheimer Disease with Convolutional Neural Networks from MRI
> The goal of this project is classifying alzheimer disease from magnetic ressonance images compressed in .nii files using convolutional neural networks architectures.

## Developers

| Full Name                          | E-mail                     |
| -----                              | ----------------           |
| `Matheus de Almeida Silva`         | ms.asilvas1@gmail.com      |
| `Matheus Augusto Somera Fernandes` | matheus.somera@gmail.com   |
| `Mauro da Silva Ribeiro`           | msr_hck1@hotmail.com       |
| `Gabriel Darin Verga`              | gabrieldarin@hotmail.com   |
| `Kaue Santos Bueno`                | kauesb@hotmail.com         |
| `Mauricio Alves Bedun Junior`      | mauriciobedun@hotmail.com  |


## Usage
See how simple that is to use this service.

**1.** Clone this repository:
```bash
$ git clone https://github.com/matheus-asilva/alzheimer-deep-learning.git
$ cd alzheimer-deep-learning
```

**2.** Create environment:
```bash
$ conda create --name alzheimer-deep-learning --file requirements.txt
$ conda activate alzheimer-deep-learning
```

**3.** Weights and Biases:
We must create an account on wandb and authenticate it.
```bash
$ wandb init
```
After the authentication, register your name and then the project name as `alzheimer-dl`

**4.** After the authentication, run
```bash
$ tasks/run_vgg.sh
```
