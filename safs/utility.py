import os
import sys
import math
import torch
import random
import utility
from enum import Enum
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data
import settings as settings
from prettytable import PrettyTable
import torch.nn.utils.prune as prune
import torch.optim.lr_scheduler as tols
from torchvision.datasets import MNIST
import torchvision.datasets as datasets
from timm.scheduler import create_scheduler
from models.localvit import create_args_localvit
import coustum_dataset
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from typing import Any


def MNIST_Loaders(state, rank_id):
    """
    Dataloader for the MNIST dataset
    """
    train_transform = transforms.Compose([
        #transforms.Resize(size=32),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        #transforms.Resize(size=32),
        transforms.ToTensor(),
    ])
    utility.Set_Seed(state.args)
    args = state.args
    train_set = MNIST('./data', train=True, download=True, transform=train_transform)
    valid_num = 10000
    num_train = len(train_set) - valid_num
    train, valid = torch.utils.data.random_split(train_set, [num_train, valid_num])  
    test = MNIST('./data', train=False, download=True, transform=test_transform)
    
    if args.gpus>1 and state.kfold_enabler:#TODO:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train,
                        num_replicas=args.world_size,
                        rank=rank_id,
                        shuffle=True,
                        seed = settings.SEED
                    )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.b, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        valid,
        batch_size=args.b, shuffle=False,
         pin_memory=True)


    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=args.b, shuffle=False,
        pin_memory=True)


    kfold_data = train_set
    return kfold_data, train_loader, val_loader, test_loader

def IMGN_loader(state, rank_id):
    """
    Dataloader for the Imagenet32 dataset
    """
    utility.Set_Seed(state.args)
    args = state.args
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.RandomResizedCrop(16),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
    transform_val = transforms.Compose([
            normalize,
        ])    
    train = coustum_dataset.IMGNET16(root='./data/IMGNET16',\
         train=0, download=False)
    valid = coustum_dataset.IMGNET16(root='./data/IMGNET16',\
         train=1, download=False)
    test = coustum_dataset.IMGNET16(root='./data/IMGNET16',\
         train=1, download=False)


    if args.gpus>1 :
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train,
                        num_replicas=args.world_size,
                        rank=rank_id,
                        shuffle=True,
                        seed = settings.SEED
                    )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train, num_workers = 4,
        batch_size=args.b, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
        valid, num_workers = 4,
        batch_size=args.b, shuffle=False,
         pin_memory=True)


    test_loader = torch.utils.data.DataLoader(
        test, num_workers = 4,
        batch_size=args.b, shuffle=False,
        pin_memory=True)

    return train , train_loader, val_loader, test_loader

def CIFAR10_Loaders(state, rank_id):
    # Dataloader for the CIFAR10 dataset
    utility.Set_Seed(state.args)
    args = state.args
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root='./data/CIFAR10',\
         train=True, transform=transform_train, download=True)

    num_valid = 10000
    num_train = len(train_set) - num_valid
    train, valid = torch.utils.data.random_split(train_set, [num_train, num_valid])

    if args.gpus>1 and state.kfold_enabler :
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train,
                        num_replicas=args.world_size,
                        rank=rank_id,
                        shuffle=True,
                        seed = settings.SEED
                    )
    else:
        train_sampler = None



    train_loader = torch.utils.data.DataLoader(
        train, num_workers = 4,
        batch_size=args.b, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        valid, num_workers = 4,
        batch_size=args.b, shuffle=False,
         pin_memory=True)

    test = datasets.CIFAR10(root='./data/CIFAR10', train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test, num_workers = 4,
        batch_size=args.b, shuffle=False,
        pin_memory=True)

    kfold_data = train_set
    return kfold_data, train_loader, val_loader, test_loader


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results




def Model_Count_Parameters(model) -> None:
    # Counting the #Zero weights of any input model

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    zeros = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        if params == 0:
            zeros += 1
    print(table)
    print(f"Total # Trainable Params: {total_params}")
    print(f"Total # Zero Params: {zeros}")
    print(f"Percenage of Zero Params: {(zeros / total_params) * 100}")
    print('-' * 20)

def Create_Checkpoint(epoch, model, optimizer, loss, optim_dictionary, results):
    # Create the training checkpoint

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model,
        'optim_dictionary': optim_dictionary,
        'optimizer_state_dict': optimizer,
        'loss': loss,
        'results': results
    }
    return checkpoint

def Save_Checkpoint(state, checkpoint_name=None):
    # Save the training model
    if 'first_train' in checkpoint_name:
        torch.save(state, settings.CHECKPOINT_Load_PATH + checkpoint_name)
    else:
        torch.save(state, settings.CHECKPOINT_PATH + checkpoint_name)

def Load_Checkpoint(model, optimizer, checkpoint_name):
    # Load the training model

    if 'Model_prune' in checkpoint_name:
        checkpoint = torch.load(settings.CHECKPOINT_PATH + checkpoint_name)
    elif 'first_train' in checkpoint_name:
        checkpoint = torch.load(settings.CHECKPOINT_Load_PATH + checkpoint_name)
        for item in model.optim_dictionary.keys():
            model.optim_dictionary[item][2][0] = checkpoint['optim_dictionary'][item][2][0]
    else :
        checkpoint = torch.load(settings.CHECKPOINT_PATH + checkpoint_name)
        for item in model.optim_dictionary.keys():
            model.optim_dictionary[item][2][0] = checkpoint['optim_dictionary'][item][2][0]
        

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss

def Set_Seed(args) -> None:
    # Initialize random seed
    settings.SEED = args.random_seed
    np.random.seed(settings.SEED)
    random.seed(settings.SEED)
    torch.manual_seed(settings.SEED)
    torch.cuda.manual_seed(settings.SEED)
    torch.cuda.manual_seed_all(settings.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_logging():
    log_format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
    logging.basicConfig(stream=sys.stdout, filemode='a', level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    fh = logging.FileHandler(settings.LOG_Text)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

def Count_Parameters_In_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

def Set_Lr_Policy(state):
    # Set learning rate policy
    if state.args.training_enabler == True:
        epoch = state.args.train_epochs
    else:
        epoch = state.args.optim_epochs
    conf = state.LR_SCHEDULER_CONFIG
    if state.lr_scheduler_name == 'ReduceLROnPlateau':
        return  torch.optim.lr_scheduler.ReduceLROnPlateau(state.optimizer, mode='min',
                                                                           factor=conf['ReduceLROnPlateau']['factor'],
                                                                           patience=conf['ReduceLROnPlateau']['patience'],
                                                                           threshold=conf['ReduceLROnPlateau']['threshold'],
                                                                           cooldown=conf['ReduceLROnPlateau']['cooldown'],
                                                                           eps=conf['ReduceLROnPlateau']['eps'],
                                                                           threshold_mode=conf['ReduceLROnPlateau']['threshold_mode'],
                                                                           min_lr=conf['ReduceLROnPlateau']['min_lr'])
    elif state.lr_scheduler_name == 'Cosine':
        conf['Cosine']['T_max'] = epoch
        return tols.CosineAnnealingLR(state.optimizer, T_max=conf['Cosine']['T_max'],
                                                       eta_min=conf['Cosine']['eta_min'])
    elif state.lr_scheduler_name == 'Cosine_resnet_IMGN16':
        T_0 = epoch//4
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts\
        (state.optimizer, T_0, T_mult=1, eta_min=5e-5)
    elif state.lr_scheduler_name == 'CosineWR':
        T_0 = epoch//4
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts\
        (state.optimizer, T_0, T_mult=1, eta_min=5e-5)
    elif state.lr_scheduler_name=="Cosine_localvit":
        localvit_args = create_args_localvit(state)
        lr_scheduler, _ = create_scheduler(localvit_args, state.optimizer)
        return lr_scheduler
    
    elif state.lr_scheduler_name == 'Step':
        conf['Step']['step_size'] = epoch//4
        return tols.StepLR(state.optimizer, step_size=conf['Step']['step_size'],
                                            gamma=conf['Step']['gamma'])
    
    elif state.lr_scheduler_name == 'Linear':
        return tols.LinearLR(state.optimizer, start_factor=conf['Linear']['start_factor'],
                                              total_iters=conf['Linear']['total_iters'])
    
    elif state.lr_scheduler_name == 'VGG_lr_scheduler':
        lambda1 = lambda epoch_: 0.5 ** (epoch_//20)
        return torch.optim.lr_scheduler.LambdaLR(state.optimizer, lr_lambda=lambda1)
    elif state.lr_scheduler_name =='Constant_Lenet':
        return tols.ConstantLR(state.optimizer, factor=1,
                                                total_iters=conf['Constant']['total_iters'])

def Show_Gradients(model):
    # Print True/False graients of the input model parameters

    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")
        
def set_requires_grad(state):
    for name, param in state.model.named_parameters():
        param.requires_grad = True
    if state.requires_grad_enabler :
        maping = state.requires_grad_data["enable"]
    else :
        maping = state.requires_grad_data["disable"]

    keys = list(state.model.optim_dictionary.keys())
    model_name = state.MODEL_ARCHITECTURES[state.args.m]
    for key in keys:
        num_active = maping[state.model.optim_dictionary[key][2][0]]
        index=0
        for name, param in state.model.named_parameters():
            
             #TODO:
            if model_name== "Lenet5":
                if "relu%s"%key in name :
                    if  index>=num_active:
                        param.requires_grad = False
                    index = index+1    
            if model_name== "VGG-16":
                if "features.%s.activation"%key in name :
                    if  index>=num_active:
                        param.requires_grad = False
                    index = index+1  
                 
             #localvit  
            if model_name== "LocalVit":     
                i, j = key
                if "blocks.%s.conv.conv.%s.activation.p"%(i, 3*j+2) in name:
                    if  index>=num_active:
                        param.requires_grad = False
                    index = index+1  
                
                
            #resnet
            if model_name== "ResNet-18":  

                if key == 'active' and "active.activation.p" in name:
                        if  index>=num_active:
                            param.requires_grad = False
                        index = index+1 
                        
                elif 'layer' in name and '.activation.p' in name and len(key)==4:    
                    i, j, k ,z = key 
                    if '%s.%s.%s.%s.activation.p'%(i, j, k, z) in name :
                        if  index>=num_active:
                            param.requires_grad = False
                        index = index+1  
                
            
            ##eff net
            if model_name== "EfficientNet_B0": 
                if key=="active" and "active.activation." in name and not "layers" in name: 
                        if  index>=num_active:
                            param.requires_grad = False
                        index = index+1  
                        
                elif "active" in name and "layers" in name and len(key)==2:
                    i, j = key
                    if "layers.%s.%s.activation.p"%(i, j) in name:
                        if  index>=num_active:
                            param.requires_grad = False
                        index = index+1  
                elif "activation" in name and "layers" in name and "features" in name and len(key)==3:
                    i, j, k = key
                    if "layers.%s.%s.%s.activation.p"%(i, j, k) in name:
                        if  index>=num_active:
                            param.requires_grad = False
                        index = index+1  
            
            
            
            
            
            
            
# TODO:
def print_weight(model, list_name=['net.0', 'net.4', 'net.9', 'net.11', 'net.13']):#TODO:
    total_zero = 0
    total_weight = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            if name in list_name:
                total_zero = total_zero + float(torch.sum(module.weight == 0))
                total_weight = total_weight + float(module.weight.nelement())
                print(
                    "Sparsity in " + name + ": {:.2f}%".format(
                        100. * float(torch.sum(module.weight == 0))
                        / float(module.weight.nelement())
                    ))

    print(
        "Sparsity in total: {:.2f}%".format(
            100. * total_zero
            / float(total_weight)
        ))



#TODO:
def Apply_pruning(state):
    # index = [n for n, mo in enumerate(model.net) if 'Linear' in str(mo) or 'SubnetConv' in str(mo)]
    list_module = []
    model_name = state.MODEL_ARCHITECTURES[state.args.m]
    if model_name == 'Lenet5':
        list_module=['conv1','conv2','fc1','fc2','fc3']
        for name, module in state.model.named_modules():
            if name in list_module:
                print(name)
                if state.args.pruning_method == 'LWM':
                    prune.l1_unstructured(module, name='weight', amount=state.args.pruning_rate)
                elif state.args.pruning_method == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
    if model_name == 'LocalVit':
        for index in range(12):
            list_module += ["blocks.%s.attn.proj"%index]
            list_module += ["blocks.%s.attn.qkv"%index]
            list_module += ["blocks.%s.conv.conv.0"%index]
            list_module += ["blocks.%s.conv.conv.3"%index]
            list_module += ["blocks.%s.conv.conv.6"%index]
        for name, module in state.model.named_modules():
            if name in list_module:
                print(name)
                if state.args.pruning_method == 'LWM':
                    prune.l1_unstructured(module, name='weight', amount=state.args.pruning_rate)
                elif state.args.pruning_method == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
    elif model_name == 'EfficientNet_B0':
        list_module = ['conv1']
        list_module += ['linear']
        for i in range(16):
                list_module += ["layers."+str(i)+".fc1"]
                list_module += ["layers."+str(i)+".fc1"]
                list_module += ["layers."+str(i)+".features.0"]
                list_module += ["layers."+str(i)+".features.3"]
                list_module += ["layers."+str(i)+".features.6"]
        for name, module in state.model.named_modules():
            if name in list_module:
                print(name)
                if state.args.pruning_method == 'LWM':
                    prune.l1_unstructured(module, name='weight', amount=state.args.pruning_rate)
                elif state.args.pruning_method == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)

    elif model_name == 'ResNet-18':
        list_module = ['conv1', 'layer1.0.features.0', 'layer1.0.features.3',\
            'layer1.1.features.0','layer1.1.features.3','layer2.0.features.0',\
            'layer2.0.features.3','layer2.1.features.0','layer2.1.features.3',\
            'layer2.0.shortcut.0','layer3.0.shortcut.0','layer4.0.shortcut.0',\
            'layer3.0.features.0', 'layer3.0.features.3', 'layer4.1.features.3',\
            'layer3.1.features.0','layer3.1.features.3','layer4.0.features.0',\
            'layer4.0.features.3','layer4.1.features.0', 'linear']
        for name, module in state.model.named_modules():
            if name in list_module:
                print(name)
                if state.args.pruning_method == 'LWM':
                    prune.l1_unstructured(module, name='weight', amount=state.args.pruning_rate)
                elif state.args.pruning_method == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)  

    elif model_name == 'VGG-16':
        index = [i for i, name in enumerate(state.model.features) if 'Conv2d' in str(name)]
        for i in index:
            list_module.append('features.' + str(i))
        list_module.append('classifier')
        for name, module in state.model.named_modules():
            if name in list_module:
                print(name)
                if state.args.pruning_method == 'LWM':
                    prune.l1_unstructured(module, name='weight', amount=state.args.pruning_rate)
                elif state.args.pruning_method == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)

    elif state.MODEL_ARCHITECTURES[state.args.m] == 'Lenet5':
        list_module = ['net.0', 'net.4', 'net.9', 'net.11', 'net.13']
        for name, module in state.model.named_modules():
            if name in list_module:
                print(name)
                if settings.Pruning_Type == 'LWM':
                    prune.l1_unstructured(module, name='weight', amount=state.args.pruning_rate)
                elif settings.Pruning_Type == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)

def Prepare_Model_secound_method(state):  # TODO:
    Apply_pruning(state)
    checkpoint_name = "prune%s.pth"%(state.MODEL_ARCHITECTURES[state.args.m])
    state.Save_Checkpoint(state.Create_Checkpoint(0), checkpoint_name)

def Pruned_Model_Fraction(model, result_dir, verbose=True):
    """
        Find pruning raio per layer. Return average of them.
        Result_dict should correspond to the checkpoint of model.
    """

    # load the dense models
    path = os.path.join(result_dir, "checkpoint_dense.pth.tar")

    pl = []

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for i, v in model.named_modules():
            if isinstance(v, (nn.Conv2d, nn.Linear)):
                if i + ".weight" in state_dict.keys():
                    d = state_dict[i + ".weight"].data.cpu().numpy()
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)
                    if verbose:
                        print(i, v, p)
        return np.mean(pl)

def Set_Prune_Rate(model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(prune_rate)

def Count_Zero_Weights(model):#TODO:
    return sum([torch.sum(i.weight == 0) for i in model.features if 'Conv2' in str(i)])

def Count_nonZero_Weights(model):#TODO:
    return sum([i.weight.nelement() for i in model.features if 'Conv2' in str(i)])

