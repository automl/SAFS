import os
import torch
import time
import logging
import argparse
import torch.nn as nn
import utility as utility
import torch.optim as optim
import settings as settings
from pruning.pruning_lib import metafire, fixed_activation, second_stage, run_fig6
from models.model import Model_Architecture

parser = argparse.ArgumentParser(description='PyTorch Activation Function Optimization')
###################################       multi  gpu      #################################
parser.add_argument('-n', '--nodes',
                    default=1,
                    type=int,
                    metavar='N')
parser.add_argument('-g', '--gpus',
                    default=1,
                    type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr',
                    default=0,
                    type=int,
                    help='ranking within the nodes')
parser.add_argument('--hpo_steps',
                    default=20,
                    type=int,
                    help='HPO')
parser.add_argument('--activ_search_steps',
                    default=20, 
                    type=int,
                    help='AF search')
###################################   Train and Evauation   ###############################

parser.add_argument('--first_train', default=1, type=int, choices=(0, 1),
                    help='If First_train {0:False, 1:True}')
parser.add_argument('--Train_after_prune', default=1, type=int, choices=(0, 1),
                    help='If Train_after_prune needed {0:False, 1:True}')
parser.add_argument('--first_stage', default=1, type=int, choices=(0, 1),
                    help='If First_stage needed {0:False, 1:True}')
parser.add_argument('--Second_stage', default=1, type=int, choices=(0, 1),
                    help='If Second_stage needed {0:False, 1:True}')
parser.add_argument('--run_activation', default='relu', type=str, choices=('relu', 'swish',\
     'TanhSoft-1', "srs", "acon", "selu", "gelu", "relu6"),
                    help='If First_train {0:False, 1:True}')
parser.add_argument('--training_enabler', default=1, type=int, choices=(0, 1),
                    help='alpha and beta train or not {0:False, 1:True}')
parser.add_argument('--run_activation_ab_flag', default=0, type=int, choices=(0, 1),
                    help='alpha and beta train or not {0:False, 1:True}')
parser.add_argument('--set_device', default=0, type=int,
                    help='set gpu device {0:gpu0, 1:gpu1}')
parser.add_argument('--num_folds', default=3, type=int,
                    help='set num_folds')
parser.add_argument('--runing_mode', default="metafire", type=str,
                    choices=("fixed_activation","second_stage", "metafire"),
                    help='choose one of runing_mode in pruning_lib')
###################################   Train and Evauation   ###############################
parser.add_argument('-d', type=int, default=0, choices=(0, 1, 2),
                    help='datasets = {0:MNIST, 1:CIFAR10, 2:Imagenet16}. MNIST could only be trained on LeNet5')
parser.add_argument('--train_epochs', default=50, type=int, metavar='N',
                    help='number of epochs to train the model')
parser.add_argument('--b', '--batch-size', default=256, type=int,
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
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', dest='gpu', action='store_true', default=True,
                    help='use gpu')
###################################  AF Optimization   ###################################
parser.add_argument('--o', '--optim-method', default=0, type=int, choices=(0, 1, 2, 3),
                    help='optimization Method {0:LAHC, 1:RS, 2:GA, 3:SA} (default: 0)'),
parser.add_argument('--m', '--model_arch', default=1, type=int, choices=(1, 2, 3, 4),
                    help='model architecture {1:Lenet5, 2:VGG-16, 3:ResNet-18, 4:EfficientNet_B0} (default: 0)')
parser.add_argument('--optim_mode', default=1, type=int, choices=(0, 1, 2),
                    help='optimization mode {0:Node, 1:Layer, 2:Network}')
parser.add_argument('--optim_epochs', default=50, type=int, metavar='N',
                    help='number of epochs to optimize AFs')
parser.add_argument('--random_seed', default=42, type=int,
                    help='manual seed for random number generator')
###################################   Pruning   ###################################
parser.add_argument('--network_type', default='Pruned', choices=("Dense", "Pruned"),
                    help='optimizing activation functions for "Dense" or "Pruned" networks')
parser.add_argument('--pruning_rate', default=0.99, type=float,
                    help='pruning rate')
parser.add_argument('--pruning_method', default='LWM', type=str, choices=("LWM"),
                    help='pruning algorithm')
parser.add_argument("--freeze_bn", action="store_true", default=False,
                    help="freeze batch-norm parameters in pruning", )
parser.add_argument("--scores_init_type", default='kaiming_normal',
                    choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"),
                    help="Which init to use for relevance scores")

def Main():
############################ args, state and random seed ###########################
    global args
    args = parser.parse_args()
    utility.set_logging()
    settings.SEED = args.random_seed
    utility.Set_Seed(args)
    logging.info('Set Random Seed to: "' + str(settings.SEED) + '"')
    
    state = settings.State(args)
#################################      multi gpu       #############################
    #multi_gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8080'
    args.b = args.b // args.gpus 
    args.world_size = args.gpus * args.nodes
#################################    crate folders     #############################
    save_id = str(int(time.time()))
    dir = 'saved_checkpoints/' + 'result_' + save_id 
    os.makedirs(dir)
    CHECKPOINT_PATH = 'saved_checkpoints/result_' + save_id + '/'
    CHECKPOINT_dense_PATH = 'saved_checkpoints/dense/'
    dir2 = CHECKPOINT_PATH + 'K_Fold' 
    os.makedirs(dir2)
    if not os.path.exists(CHECKPOINT_dense_PATH):
        os.makedirs(CHECKPOINT_dense_PATH)
        os.makedirs(CHECKPOINT_dense_PATH+"/K_Fold")
        os.makedirs(CHECKPOINT_dense_PATH+"/K_Fold/best")
############################       set gpu       ##################################
    if torch.cuda.is_available() and args.gpu:
        torch.cuda.set_device(args.set_device)
############################    set model arch.     ###############################
    model = Model_Architecture(state)
##############################       criterior      ###############################
    criterior = nn.CrossEntropyLoss()

    if torch.cuda.is_available() and args.gpu:
        model.cuda()
        criterior.cuda()
    else:
        model.cpu()
        criterior.cpu()
    if state.MODEL_ARCHITECTURES[args.m] == "Lenet5":
        state.Optim_default_params['learning_rate'] = 0.001
        state.Optim_params['learning_rate'] = 0.001
        state.optimizer_name = "Adam"
    if state.MODEL_ARCHITECTURES[args.m] == "ResNet-18" and args.d == 0:
        state.Optim_default_params['learning_rate'] = 0.001
        state.Optim_params['learning_rate'] = 0.001
        state.optimizer_name = "Adam"
    if state.MODEL_ARCHITECTURES[args.m] == "ResNet-18" and args.d == 1:
        state.Optim_default_params['learning_rate'] = 0.01
        state.Optim_params['learning_rate'] = 0.001
        state.optimizer_name = "SGD"
    elif state.MODEL_ARCHITECTURES[args.m] == "ResNet-18" and args.d == 2:
        state.Optim_params['learning_rate'] = 0.1
        state.Optim_params['wd'] = 2e-05
        state.Optim_default_params['learning_rate'] = 0.01
        state.optimizer_name = "SGD"
    elif state.MODEL_ARCHITECTURES[args.m] == 'VGG-16':
        state.Optim_default_params['learning_rate'] = 0.001
        state.Optim_params['learning_rate'] = 0.001
        state.optimizer_name = "SGD"
    elif state.MODEL_ARCHITECTURES[args.m] == 'LocalVit':
        state.Optim_default_params['learning_rate'] = 5e-4
        state.Optim_params['learning_rate'] = 5e-4
        state.optimizer_name = "localvit"
    elif state.MODEL_ARCHITECTURES[args.m] == 'EfficientNet_B0':
        state.Optim_default_params['learning_rate'] = 0.001
        state.Optim_params['learning_rate'] = 0.001
        state.optimizer_name = "SGD"
        

    ############################   set state paaram.1 #############################        
    state.model = model
    state.criterior = criterior
    state.prepare_dataset(state,0)#rankid
    state.update_optimizer()
    ######################## Config LearningRate_SCHEDULER ########################
    if state.MODEL_ARCHITECTURES[args.m] == "Lenet5":
        lr_scheduler_name = 'Constant_Lenet'
    elif state.MODEL_ARCHITECTURES[args.m] == "ResNet-18" and args.d == 0:
        lr_scheduler_name = 'cc'
    elif state.MODEL_ARCHITECTURES[args.m] == "ResNet-18" and args.d == 1:
        lr_scheduler_name = 'ReduceLROnPlateau'
    elif state.MODEL_ARCHITECTURES[args.m] == "ResNet-18" and args.d == 2:
        lr_scheduler_name = 'Cosine_resnet_IMGN16'
    elif state.MODEL_ARCHITECTURES[args.m] == 'VGG-16':
        lr_scheduler_name = 'VGG_lr_scheduler'
    elif state.MODEL_ARCHITECTURES[args.m]=="LocalVit":
        lr_scheduler_name = "Cosine_localvit"
    elif state.MODEL_ARCHITECTURES[args.m] == 'EfficientNet_B0':
        lr_scheduler_name = 'Cosine_resnet_IMGN16'
    ############################   set state paaram.2 #############################
    state.device_id = state.args.set_device
    state.lr_scheduler_name = lr_scheduler_name
    state.CHECKPOINT_PATH = CHECKPOINT_PATH
    state.CHECKPOINT_dense_PATH = CHECKPOINT_dense_PATH
    utility.Set_Lr_Policy(state)
    ############################    logging params.   #############################
    logging.info('Dataset: "' + state.DATASETS[args.d] + '"')
    logging.info('Backbone architecture: "' + state.MODEL_ARCHITECTURES[args.m] + '"')
    logging.info('Optimization method: "' + state.OPTIM_METHODS[args.o] + '"')
    logging.info('Optimization mode: "' + state.OPTIM_MODES[args.optim_mode] + '"')
    logging.info('lr_scheduler_name: "' + state.lr_scheduler_name  + '"')

    if args.runing_mode == "second_stage":
        model = run_fig6(state)
    elif args.runing_mode ==  "fixed_activation":
        model = fixed_activation(state)
    elif args.runing_mode == "metafire":
         model = metafire(state)


if __name__ == '__main__':
    Main()
    logging.info('Done!')


