# Learning Activation Functions for Sparse Neural Networks (AutoML 2023)

SAFS is a framework for designing novel activation functions for arbitrary sparse (pruned) convolutional neural networks.

SAFS  is written in Python3 and tested with Python 3.10. 


## Installation

- `git clone https://github.com/Mohammadloni/SAFS.git && cd SAFS`
- `git branch -a && git checkout multi_gpu`
- `pip install -r requirements.txt`


## Run the repository


We will use `main.py` for all our experiments on the MNIST and CIFAR-10 and Imagenet16 datasets.


- `--d`: dataset selection (0:MNIST, 1:CIFAR10, 2:Imagenet16)
- `--model_arch`: model selection(1:Lenet5, 2:VGG-16, 3:ResNet-18, 4:EfficientNet_B0)
- `--optim-method`: stage1 search strategy(0:LAHC, 1:RS, 2:GA, 3:SA)
- `--first_train`: enable first_train(True, False)
- `--Train_after_prune'`: enable Train_after_prune(True, False)
- `--first_stage'`: enable first_stage(True, False)
- `--Second_stage`: enable Second_stage(True, False)
- `--gpus`: num of available gpus
- `--pruning_method`: LWM
- `--set_device`: select a gpu(0, 1, 2, ...)


### Runing a simple example

First, we train VGG-16 networks `--first_train=1` then pruned the model  with `pruning_rate = 0.99`. Then, retrain the pruned model `--Train_after_prune=1` after that we desing new activation functions for each layer of the network using the LAHC algorithm `--first_stage=1`. Finaly we run the Second stage HPO `--Second_stage=1`.  

`python main.py --model_arch=2 --runing_mode="metafire" --pruning_rate=0.99 --first_train=1  --Train_after_prune=1 --first_stage=1 --Second_stage=1`



### Summery of results
![alt text](./docs/images/results_table.png?raw=true)


## Contacting us

If you have trouble using SAFS, a concrete question or found a bug, please create an [issue](https://github.com/Mohammadloni/SAFS/issues). This is the easiest way to communicate about these things with us. 

For all other inquiries, please write an email to mohammad.loni@mdu.se.

Copyright (C) 2022-2023  [AutoML Group](http://www.automl.org).
