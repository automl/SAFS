import os
import sys
import torch
import time
import logging
import argparse
import torch.nn as nn
import utility as utility
import torch.optim as optim
import settings as settings
import torch.multiprocessing as mp
import training.train_lib as train_lib
from pruning.pruning_lib import metafire, pruning_test,\
                                pruning_test_kfold , optuna_runing
from optimization.hpo_lib import *
from models.model import Model_Architecture

#python main.py --model_arch=2 --set_device=0 --First_train=0 --pruning_rate=0.99 --run_activation="swish" --run_activation_ab_flag=0 --run_mode=0 
#python main.py --model_arch=2 --set_device=0 --runing_mode="optuna_runing" --First_train=0 --pruning_rate=0.99  
#python main.py --model_arch=2 --set_device=1 --runing_mode="MetaFire" --First_train=0 --pruning_rate=0.99  --Train_after_prune=1 --First_stage=1 --Second_stage=1

parser = argparse.ArgumentParser(description='PyTorch Activation Function Optimization')


###################################       multi  gpu      #################################
#TODO: multi_gpu
parser.add_argument('-n', '--nodes',
                    default=1,
                    type=int,
                    metavar='N')
parser.add_argument('-g', '--gpus',
                    default=2,
                    type=int,
                    help='number of gpus per node')
 
###################################   Train and Evauation   ###############################

parser.add_argument('--First_train', default=0, type=int, choices=(0, 1),
                    help='If First_train {0:False, 1:True}')

parser.add_argument('--Train_after_prune', default=0, type=int, choices=(0, 1),
                    help='If Train_after_prune needed {0:False, 1:True}')

parser.add_argument('--First_stage', default=0, type=int, choices=(0, 1),
                    help='If First_stage needed {0:False, 1:True}')

parser.add_argument('--Second_stage', default=0, type=int, choices=(0, 1),
                    help='If Second_stage needed {0:False, 1:True}')

parser.add_argument('--run_activation', default='relu', type=str, choices=('relu', 'swish',\
     'TanhSoft-1', "srs", "acon", "selu", "gelu", "relu6"),
                    help='If First_train {0:False, 1:True}')

parser.add_argument('--run_activation_ab_flag', default=0, type=int, choices=(0, 1),
                    help='alpha and beta train or not {0:False, 1:True}')

parser.add_argument('--set_device', default=0, type=int,
                    help='set gpu device {0:gpu0, 1:gpu1}')

parser.add_argument('--runing_mode', default="MetaFire", type=str,
                    choices=("const_activation_kfold", "optuna_runing", "MetaFire"),
                    help='choose one of runing_mode in pruning_lib')
###################################   Train and Evauation   ###############################
parser.add_argument('-d', type=int, default=0, choices=(0, 1),
                    help='datasets = {0:MNIST, 1:CIFAR10}. MNIST could only be trained on LeNet5')
parser.add_argument('--train_epochs', default=200, type=int, metavar='N',
                    help='number of epochs to train the model')
parser.add_argument('--b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--save_id', default='default', type=str)

parser.add_argument('--lr_scheduler', default='ReduceLROnPlateau', type=str,
                    choices=("ReduceLROnPlateau", "Constant", "Cosine", "Step", "Linear"),
                    help='learning rate scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', dest='gpu', action='store_true', default=True,
                    help='use gpu')
###################################  AF Optimization   ###################################
parser.add_argument('--o', '--optim-method', default=0, type=int, choices=(0, 1, 2, 3, 4),
                    help='optimization Method {0:LAHC, 1:RS, 2:GA, 3:SA, 4:Random_Assignment} (default: 0)'),
parser.add_argument('--m', '--model_arch', default=2, type=int, choices=(0, 1, 2, 3),
                    help='model architecture {0:One-Layer MLP, 1:Lenet5, 2:VGG-16, 3:ResNet-18, 4:EfficientNet_B0} (default: 0)')
parser.add_argument('--optim_mode', default=1, type=int, choices=(0, 1, 2),
                    help='optimization mode {0:Node, 1:Layer, 2:Network}')
parser.add_argument('--optim_epochs', default=80, type=int, metavar='N',
                    help='number of epochs to optimize AFs')
parser.add_argument('--random_seed', default=True,
                    help='manual seed for random number generator')
###################################   Pruning   ###################################
parser.add_argument('--network_type', default='Pruned', choices=("Dense", "Pruned"),
                    help='optimizing activation functions for "Dense" or "Pruned" networks')
parser.add_argument('--pruning_rate', default=0.99, type=float,
                    help='pruning rate')
parser.add_argument('--pruning_method', default='LWM', type=str, choices=("LWM", "Hydra"),
                    help='pruning algorithm')
parser.add_argument("--freeze_bn", action="store_true", default=False,
                    help="freeze batch-norm parameters in pruning", )
parser.add_argument("--scores_init_type", default='kaiming_normal',
                    choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"),
                    help="Which init to use for relevance scores")

def Main():

    global args
    args = parser.parse_args()
    log_format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
    logging.basicConfig(stream=sys.stdout, filemode='a', level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    if not os.path.exists(settings.LOG_Text):
        os.mkdir(settings.LOG_DIR)
    fh = logging.FileHandler(settings.LOG_Text)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
#####################################################################
#TODO: multi_gpu
    args.world_size = args.gpus * args.nodes
    args.batch_size = args.batch_size // args.gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8080'
#####################################################################

    settings.First_train = args.First_train
    settings.Train_after_prune = args.Train_after_prune
    settings.First_stage = args.First_stage
    settings.Second_stage = args.Second_stage
    settings.run_activation_ab_flag = args.run_activation_ab_flag
    settings.run_activation = args.run_activation
    settings.set_device = args.set_device

#####################################################################
    settings.LOGGING = logging
    settings.save_id = str(time.time())
    dir = settings.CHECKPOINT_PATH + 'result_' + settings.save_id 
    os.makedirs(dir)
    settings.CHECKPOINT_PATH = settings.CHECKPOINT_PATH + 'result_' + settings.save_id + '/'
    dir2 = settings.CHECKPOINT_PATH + 'K_Fold' 
    os.makedirs(dir2)

    settings.GPU_ENABLED = args.gpu
    if torch.cuda.is_available() and settings.GPU_ENABLED:
        torch.cuda.set_device(settings.set_device)

    # settings.DATASET = args.d
    logging.info('Dataset: "' + settings.DATASETS[settings.DATASET] + '"')

    settings.MODEL_ARCHITECTURE = args.m
    logging.info('Backbone architecture: "' + settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '"')
    logging.info('Target network is: "' + args.network_type + '"')

    settings.PRUNED_DENSE = args.network_type

    if (settings.PRUNED_DENSE == "Pruned"):
        logging.info('pruning method %s pruning ration %f', args.pruning_method, args.pruning_rate)
        logging.info('Pruning method: "' + args.pruning_method + '"')
    settings.Pruning_Rate = args.pruning_rate
    settings.OPTIM_METHOD = args.o
    logging.info('Optimization method: "' + settings.OPTIM_METHODS[settings.OPTIM_METHOD] + '"')

    settings.OPTIM_MODE = args.optim_mode
    logging.info('Optimization mode: "' + settings.OPTIM_MODES[settings.OPTIM_MODE] + '"')

    settings.TRAIN_EPOCH = args.train_epochs
    settings.OPTIM_EPOCH = args.optim_epochs
    settings.BATCH_SIZE = args.b
    settings.WORKERS = args.workers
    if args.resume != None:
        settings.CHECKPOINT = True
        settings.CHECKPOINT_PATH = args.resume
        if not os.path.exists(settings.CHECKPOINT_PATH):
            os.mkdir(settings.CHECKPOINT_PATH)

    ########################      set model arch.       ########################
    model = Model_Architecture()
    if torch.cuda.is_available() and settings.GPU_ENABLED:
        model.cuda()
    else:
        model.cpu()
    ########################      set random_seed        ########################
    if args.random_seed:
        logging.info('Set Random Seed to: "' + str(settings.SEED) + '"')
        utility.Set_Seed()
    ######################## set criterior and optimizer ########################
    criterior = nn.CrossEntropyLoss()
    if torch.cuda.is_available() and settings.GPU_ENABLED:
        criterior = criterior.cuda()
    else:
        criterior = criterior.cpu()

    for name, param in model.named_parameters():
        param.requires_grad = True
        if 'alpha' in name or 'beta' in name:
            param.requires_grad = False
            
    if settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == "ResNet-18":
        settings.Optim_default['learning_rate'] = 0.01
    elif settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == 'VGG-16':
        settings.Optim_default['learning_rate'] = 0.001

    optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                          momentum=settings.Optim_default['momentum'],
                          weight_decay=settings.Optim_default['wd'])

    ######################## Config LearningRate_SCHEDULER ########################
    if settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == "ResNet-18":
        settings.lr_scheduler_name = 'ReduceLROnPlateau'
    elif settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == 'VGG-16':
        settings.lr_scheduler_name = 'VGG_lr_scheduler'
    utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)

    ########################       prepare datasets      ########################
    if settings.DATASET == 0:
        train_loader, test_loader = utility.MNIST_Loaders(settings.BATCH_SIZE, settings.WORKERS)
    else:
        all_dataset, train_loader, test_loader, final_test_loader = utility.CIFAR10_Loaders(settings.BATCH_SIZE, settings.WORKERS)
    ##################### store global setting and variable #####################
    settings.DATA_Test = test_loader
    settings.DATA_Train = train_loader
    settings.All_Dataset = all_dataset
    settings.DATA_Final_Test = final_test_loader
    settings.Criterior = criterior
    settings.Optimizer = optimizer

    if args.runing_mode == "optuna_runing":
        model = optuna_runing(settings.Pruning_Rate ,model, args, criterior, optimizer, train_loader, test_loader)
    elif args.runing_mode == "const_activation_kfold":
        model = pruning_test_kfold(settings.Pruning_Rate ,model, args, criterior, optimizer, train_loader, test_loader)
    elif args.runing_mode == "MetaFire":
        model = metafire(settings.Pruning_Rate ,model, args, criterior, optimizer, train_loader, test_loader)


if __name__ == '__main__':
    Main()
    logging.info('Done!')


