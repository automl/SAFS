import torch
import random
import os
import numpy as np
import torch.nn as nn
from torchvision.datasets import MNIST

import settings as settings
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from prettytable import PrettyTable
import torch.optim.lr_scheduler as tols
import math
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset



def MNIST_Loaders(batch_size, num_workers):
    """
    Dataloader for the MNIST dataser
    """

    train = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    valid_num = int(0.1*len(train))
    num_train = len(train) - valid_num
    train, valid = torch.utils.data.random_split(train, [num_train, valid_num])
    test = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

    cuda = torch.cuda.is_available()

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda \
        else {}  # dict(shuffle=True, batch_size=64)
    train_loader = dataloader.DataLoader(train, **dataloader_args)
    valid_loader = dataloader.DataLoader(valid, **dataloader_args)
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, valid_loader, test_loader


def CIFAR10_Loaders(args, rank_id, batch_size, workers):
    # Dataloader for the CIFAR10 dataset

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


    num_valid = int(0.1*len(train_set))
    num_train = len(train_set) - num_valid
    train, valid = torch.utils.data.random_split(train_set, [num_train, num_valid])

#TODO: multi_gpu
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train,
        num_replicas=args.world_size,
        rank=rank_id,
        shuffle=True
    )


    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        valid,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    test = datasets.CIFAR10(root='./data/CIFAR10', train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    # all_data = ConcatDataset([train, test])
    all_data = train_set
    return all_data, train_loader, val_loader, test_loader


def MLP_Count_Parameters(model) -> None:
    # Counting the #Zero weights of a one-layer MLP model

    nonzeros = 0
    total_params = 0

    for i in range(len(model.layers[0].weight)):
        for k in range(len(model.layers[0].weight[i].data)):
            nonzeros = nonzeros + torch.count_nonzero(model.layers[0].weight[i].data[k]).item()
            for j in range(torch.numel(model.layers[0].weight[i].data[k])):
                # print (model.layers[0].weight[i].data[k])
                total_params = total_params + 1

    print(f"total_params, non-zeros: {total_params, nonzeros}")

    print(f"Percentage of zero Params: {((total_params - nonzeros) / total_params) * 100}")
    print('-' * 20)


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


def Load_Checkpoint(model, optimizer, checkpoint_name=''):
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


def Set_Seed() -> None:
    # Initialize random seed
    np.random.seed(settings.SEED)
    random.seed(settings.SEED)
    torch.manual_seed(settings.SEED)
    torch.cuda.manual_seed(settings.SEED)
    torch.cuda.manual_seed_all(settings.SEED)


def Count_Parameters_In_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def Set_Lr_Policy(lr_scheduler, optimizer):
    # Set learning rate policy

    conf = settings.LR_SCHEDULER_CONFIG
    if lr_scheduler == 'ReduceLROnPlateau':
        settings.LR_SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=conf['ReduceLROnPlateau']['factor'],
                                                                           patience=conf['ReduceLROnPlateau']['patience'],
                                                                           threshold=conf['ReduceLROnPlateau']['threshold'],
                                                                           cooldown=conf['ReduceLROnPlateau']['cooldown'],
                                                                           eps=conf['ReduceLROnPlateau']['eps'],
                                                                           threshold_mode=conf['ReduceLROnPlateau']['threshold_mode'],
                                                                           min_lr=conf['ReduceLROnPlateau']['min_lr'])
    elif lr_scheduler == 'Cosine':
        settings.LR_SCHEDULER = tols.CosineAnnealingLR(optimizer, T_max=conf['Cosine']['T_max'],
                                                       eta_min=conf['Cosine']['eta_min'])
    elif lr_scheduler == 'Step':
        settings.LR_SCHEDULER = tols.StepLR(optimizer, step_size=conf['Step']['step_size'],
                                            gamma=conf['Step']['gamma'])
    elif lr_scheduler == 'Linear':
        settings.LR_SCHEDULER = tols.LinearLR(optimizer, start_factor=conf['Linear']['start_factor'],
                                              total_iters=conf['Linear']['total_iters'])
    elif lr_scheduler == 'VGG_lr_scheduler':
        # settings.LR_SCHEDULER = VGG_lr_scheduler(optimizer, settings.Optim_default['learning_rate'])
        lambda1 = lambda epoch: 0.5 ** (epoch//20)
        settings.LR_SCHEDULER = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    else:
        settings.LR_SCHEDULER = tols.ConstantLR(optimizer, factor=conf['Constant']['factor'],
                                                total_iters=conf['Constant']['total_iters'])


# lambda1 = lambda epoch: 0.65 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
class VGG_lr_scheduler:
    def __init__(self, optimizer, lr=0.001):
        self.lr = lr
        self.optimizer = optimizer

    def __call__(self, num_update):
        return  self.optimizer

    def step(self, num_update):
        self.optimizer.param_groups[0]["lr"] = self.lr * (0.5 ** (num_update // 20))



def Show_Gradients(model):
    # Print True/False graients of the input model parameters

    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")


def Freeze_Vars(model, var_name, freeze_bn=False):
    # Freeze vars. If freeze_bn then only freeze batch_norm parameters

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def Unfreeze_Vars(model, var_name):
    # Unfreeze vars.

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def Initialize_Scores(model, init_type):
    # Initialize scores

    print(f"Initialization relevance score with {init_type} initialization")
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )


def Prepare_Model(model, args, training_mode):
    '''
    1. Set model pruning rate
    2. Set gradients base on training mode.
    3. Initialize scores
    '''

    if training_mode == "Pretrain":
        print(f"#################### Pre-training network ####################")
        print("training weights only")
        Set_Prune_Rate(model, 1.0)
        Freeze_Vars(model, "popup_scores", args.freeze_bn)
        Unfreeze_Vars(model, "weight")
        Unfreeze_Vars(model, "bias")

    elif training_mode == "Prune":
        print(f"#################### Pruning network ####################")
        print("training importance scores only")
        Set_Prune_Rate(model, 1 - args.pruning_rate)
        Unfreeze_Vars(model, "popup_scores")
        Freeze_Vars(model, "weight", args.freeze_bn)
        Freeze_Vars(model, "bias", args.freeze_bn)

    elif training_mode == "Finetune":
        print(f"#################### Fine-tuning network ####################")
        print(" fine-tuning important weigths only")
        Set_Prune_Rate(model, 1.0)
        Freeze_Vars(model, "popup_scores", args.freeze_bn)
        Unfreeze_Vars(model, "weight")
        Unfreeze_Vars(model, "bias")

    else:
        assert False, f"{args.training_mode} mode is not supported"

    Initialize_Scores(model, args.scores_init_type)


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


import torch.nn.utils.prune as prune

global threshold
threshold = 0.25


class threshold_weights_PruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor"""
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return torch.abs(t) > threshold


def threshold_weights_unstructured(module, name):
    threshold_weights_PruningMethod.apply(module, name)
    return module
#TODO:
def Apply_pruning(model,PR_amount, list_module=['net.0', 'net.4', 'net.9', 'net.11', 'net.13']):
    # index = [n for n, mo in enumerate(model.net) if 'Linear' in str(mo) or 'SubnetConv' in str(mo)]
    list_module = []
    if settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == 'EfficientNet_B0':
        list_module = ['conv1']
        list_module += ['linear']
        for i in range(16):
                list_module += ["layers."+str(i)+".fc1"]
                list_module += ["layers."+str(i)+".fc1"]
                list_module += ["layers."+str(i)+".features.0"]
                list_module += ["layers."+str(i)+".features.3"]
                list_module += ["layers."+str(i)+".features.6"]

        for name, module in model.named_modules():
            if name in list_module:
                print(name)
                if settings.Pruning_Type == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=PR_amount)
                elif settings.Pruning_Type == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)   
    if settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == 'ResNet-18':
        list_module = ['conv1', 'layer1.0.features.0', 'layer1.0.features.3',\
            'layer1.1.features.0','layer1.1.features.3','layer2.0.features.0',\
            'layer2.0.features.3','layer2.1.features.0','layer2.1.features.3',\
            'layer2.0.shortcut.0','layer3.0.shortcut.0','layer4.0.shortcut.0',\
            'layer3.0.features.0', 'layer3.0.features.3', 'layer4.1.features.3',\
            'layer3.1.features.0','layer3.1.features.3','layer4.0.features.0',\
            'layer4.0.features.3','layer4.1.features.0', 'linear']
        for name, module in model.named_modules():
            if name in list_module:
                print(name)
                if settings.Pruning_Type == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=PR_amount)
                elif settings.Pruning_Type == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)    
    if settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == 'VGG-16':
        index = [i for i, name in enumerate(model.features) if 'Conv2d' in str(name)]
        for i in index:
            list_module.append('features.' + str(i))
        list_module.append('classifier')
        for name, module in model.named_modules():
            if name in list_module:
                print(name)
                if settings.Pruning_Type == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=PR_amount)
                elif settings.Pruning_Type == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
    elif settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == 'Lenet5':
        list_module = ['net.0', 'net.4', 'net.9', 'net.11', 'net.13']
        for name, module in model.named_modules():
            if name in list_module:
                print(name)
                if settings.Pruning_Type == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=PR_amount)
                elif settings.Pruning_Type == 'ln_structured':
                    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)


def Prepare_Model_secound_method(model,amount, args, training_mode):  # TODO:
    Apply_pruning(model,amount)


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

def Initialize_Scaled_Score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores.data = (
                    math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )