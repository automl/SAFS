import os
import time
import copy
import torch
import settings 
import numpy as np
import torch.nn as nn
import models.activation
import utility as utility
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torch.utils.data.dataloader as dataloader
from torch.nn.parallel import DistributedDataParallel as DDP

logging = settings.LOGGING 
##########################################################################multi

##########################################################################
def Training(model, mode, device_id, criterior, optimizer ,train_loader):

    if settings.TRAINING_ENABLER == True:
        epoch = settings.TRAIN_EPOCH
    else:
        epoch = settings.OPTIM_EPOCH

    logging = settings.LOGGING

    acc_best = 0
    results = []
    utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)
    scheduler = settings.LR_SCHEDULER

    for i in range(epoch):
        start_time = time.time()
        train_loss, acc, total_checked = Train_Model(model, criterior, optimizer, train_loader)
        end_time = time.time()
        epoch_time = end_time - start_time
    
        test_loss, test_acc = Test_Model(model, criterior, settings.DATA_Test)
        lr = optimizer.param_groups[0]['lr']
        #TODO: enable scheduler
        if settings.Enable_scheduler:
            if settings.lr_scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(train_loss)  
            else:
                scheduler.step()

        if acc_best < acc:
            acc_best = acc
            state_ = model.state_dict()

        mean_, std_ = calc_grad_flow(model.named_parameters())
        total_grad_mean = np.zeros(len(mean_))
        total_grad_std = np.zeros(len(std_))
        total_grad_mean = [gbase1 + g1 * float(total_checked) for gbase1, g1 in zip(total_grad_mean, mean_)]
        total_grad_std = [gbase1 + g1 * float(total_checked) for gbase1, g1 in zip(total_grad_std, std_)]

        total_grad_mean = [gbase1 / total_checked for gbase1 in total_grad_mean]
        total_grad_std = [gbase1 / total_checked for gbase1 in total_grad_std]

        logging.info("Device {}epoch_{} time:{} lr:{} train_acc:{} test_acc:{}".format(device_id,\
             i, epoch_time, lr, acc, test_acc))
       
        results.append({"train_acc":(train_loss, acc), "test_acc1":(test_loss, test_acc),\
                        "grad_flow":(total_grad_mean,total_grad_std), "epoch_time":epoch_time})

    if mode == 'No_Save':
        pass
    else:
        checkpoint = utility.Create_Checkpoint(i, state_, optimizer.state_dict(), train_loss, model.optim_dictionary, 
        {'results':results})
        utility.Save_Checkpoint(checkpoint, 'Model_' + mode + settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE]+ '.pth')
   

    model.load_state_dict(state_)
    return model

def Train_Model(model, criterior, optimizer, train_loader):
    model.train()
    i = 0
    running_loss = 0.0
    correct = 0
    total = 0
    total_checked = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available() and settings.GPU_ENABLED:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model.forwards(data)
        i = i + 1
        loss = criterior(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        total_checked += data.size(0)

    acc = 100. * correct / total
    end_time = time.time()
    running_loss /= len(train_loader)
    return running_loss, acc, total_checked

def Test_Model(model, criterior, test_loader):

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        correct_prediction = 0.0
        total_pred = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available() and settings.GPU_ENABLED:
                data = data.cuda()
                target = target.cuda()
            output = model.forwards(data)
            _, predicted = torch.max(output.data, 1)
            predicted = predicted
            total_pred += target.size(0)
            correct_prediction += (predicted == target).sum().item()
            loss = criterior(output, target).detach()
            running_loss += loss.item()
    running_loss /= len(test_loader)
    acc = (correct_prediction / total_pred) * 100.0
    return running_loss, acc

def MLP_Finetuning(model, criterior, optimizer):
    model_temp = model
    mask = copy.deepcopy(model_temp.layers[0].weight)
    mask.requires_grad = False
    Counter = 0
    for i in range(len(model_temp.layers[0].weight)):
        for k in range(len(model_temp.layers[0].weight[i].data)):
            if (abs(model_temp.layers[0].weight[i].data[k].double()) < 0.3):
                Counter = Counter + 1
                mask[i][k] = 0.0
            else:
                mask[i][k] = 1.0
    if settings.TRAINING_ENABLER == True:
        epoch=settings.TRAIN_EPOCH
    else:
        epoch=settings.OPTIM_EPOCH
    logging = settings.LOGGING
    scheduler = settings.LR_SCHEDULER
    logging.info('One-layer MLP Finetuning')
    for i in range(epoch):
        train_loss = Train_Model_Sparse(model_temp, criterior, optimizer, mask)
        test_loss, test_acc = Test_Model(model_temp, criterior)
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e test_loss %f  test_acc %f', epoch, lr, test_loss, test_acc)
        scheduler.step()
    torch.save(model_temp, 'saved_models/Model_FT.pth')
    return model_temp

def Train_Model_Sparse(model, criterior, optimizer, mask):

    if settings.DATASET == 0:
        train_loader, test_loader = utility.MNIST_Loaders(settings.BATCH_SIZE, settings.WORKERS)
    else:
        train_loader, test_loader = utility.CIFAR10_Loaders(settings.BATCH_SIZE, settings.WORKERS)

    model.train()
    i = 0
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.cuda.is_available() and settings.GPU_ENABLED:
            data = data.cuda()
            target = target.cuda()

        target = target.long()

        output = model.forwards(data)

        i = i + 1
        loss = criterior(output, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        model.layers[0].weight.data=model.layers[0].weight.data*mask
    end_time = time.time()
    running_loss /= len(train_loader)
    print('Training_loss:', running_loss, 'Time:', end_time - start_time, 's')
    return running_loss


def calc_grad_flow(named_parameters):
    layers = []
    k = 0
    res_mean = []
    res_std = []

    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and \
                ("acon_pp" not in n) and ("beta" not in n) and ("alpha" not in n):
            layers.append(n)
            res_mean.append(p.grad.abs().mean().cpu())
            res_std.append(p.grad.abs().std().cpu())
            k += 1
    res_mean = np.array(res_mean)
    res_std = np.array(res_std)
    return res_mean, res_std