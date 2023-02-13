# SAFS: Learning Activation Functions for Sparse Neural Networks


SAFS is a tool for designing novel activation functions for arbitrary dense and sparse (pruned) convolutional neural networks. The main core consists of two-stage optimization in combination with a hyper-parameter optimization method.

SAFS  is written in Python3 and tested with Python 3.7. 

## Installation
```
git clone https://github.com/automl/SAFS.git
cd SAFS
conda create -n safs python=3.8
conda activate safs

# Install for usage
pip install .

# Install for development
make install-dev
```

Documentation at https://automl.github.io/SAFS/main

## Run the Repository


We will use `train.py` for all our experiments on the MNIST and CIFAR-10 datasets.


- `--d`: cifar10, MNIST
- `--model_arch`: One-Layer MLP, Lenet5, VGG-16, ResNet-18
- `--optim-method`: LAHC, RS, GA, SA, Random_Assignment
- `--optim_mode`: Node, Layer, Network
- `--network_type`: Dense, Prune
- `--pruning_method`: LWM, Hydra
- `--First_train`: True, False
- `--Train_after_prune'`: True, False
- `--First_stage'`: True, False
- `--Second_stage`: True, False  

### Runing a simple example

First, we train VGG-16 networks `--First_train=1` then pruned the model  with `pruning_rate = 0.99`. Then, retrain the pruned model `--Train_after_prune=1` after that we desing new activation functions for each layer of the network using the LAHC algorithm `--First_stage=1`. Finaly we run the Second stage HPO `--Second_stage=1`.  

`python main.py --model_arch=2 --runing_mode="MetaFire" --pruning_rate=0.99 --First_train=1  --Train_after_prune=1 --First_stage=1 --Second_stage=1`