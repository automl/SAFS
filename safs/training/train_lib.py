import time
import torch
import numpy as np
import torch.nn as nn
import utility as utility
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torch.utils.data.dataloader as dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
import logging 

utility.set_logging()




def Training(state, mode, result_queue):

    args = state.args
    state.update_optimizer()
    optimizer = state.optimizer
    state_dict = state.model.module.state_dict()
    if args.training_enabler == True:
        epoch = args.train_epochs
    else:
        epoch = args.optim_epochs
    scheduler =  utility.Set_Lr_Policy(state)
    acc_best = 0
    results = []
    test_loss, test_acc = 1,(1,1)
    for i in range(epoch):   
        start_time = time.time()
        train_loss, acc, grad_flow = Train_Model(state)


        lr = optimizer.param_groups[0]['lr']
        end_time = time.time()
        epoch_time = end_time - start_time  


        if state.enable_scheduler:
            if state.lr_scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(train_loss)  
            elif state.lr_scheduler_name == "Cosine_localvit":
                scheduler.step(i)
            else:
                scheduler.step()

        if state.args.d == 2:
            if mode != 'No_Save':
                if True:#(i>300)or(i<300 and i>100 and i%3==0):#TODO:
                    test_loss, test_acc = Test_Model(state)
            else :
                if True:
                    test_loss, test_acc = Test_Model(state)

        else:
            if mode != 'No_Save':
                if True:
                    test_loss, test_acc = Test_Model(state)
            else :
                if True:
                    test_loss, test_acc = Test_Model(state)
        if acc_best < test_acc[0]:
                acc_best = test_acc[0]
                state_dict = state.model.module.state_dict()
        #
        if state.args.d == 2:
            if i==10:
                if acc_best<2:
                    break
                
        if state.args.set_device ==   state.device_id : 
            logging.info("Device {}epoch_{} time:{} lr:{} train_acc:{} test_acc:{}".format(state.device_id,\
                i, epoch_time, lr, acc, test_acc))

        results.append({"train_acc":(train_loss, acc), "test_acc1":(test_loss, test_acc),\
                        "grad_flow":grad_flow, "epoch_time":epoch_time})
    
    
    logging.info("max acc test: %s" %acc_best)

    state.model = state.model.module
    if mode != 'No_Save':
        ch_name = "%s.pth"%(mode) 
        
        result_queue.put((state.device_id, acc_best))
        state.Save_Checkpoint(state.Create_Checkpoint(results), ch_name)
    else:
        result_queue.put((state.device_id, acc_best))
    state.model.load_state_dict(state_dict)
    if state.args.d != 2:
        state.test_final=1
    running_loss, acc_=Test_Model(state)
    print("final_test:",Test_Model(state))
    state.test_final=0
    with open("%sfinal_test_%s.txt"%\
                (state.CHECKPOINT_PATH,state.args.o), "a") as f:
                f.write("final_test_%s acc:%s \n "%(mode, acc_))
                f.close()



def Train_Model(state):
   

    train_loader = state.data_pack["train_loader"]
    model = state.model
    device_id = state.device_id
    model.train()
    i = 0
    running_loss = 0.0
    correct = 0
    total = 0
    total_checked = 0
    grad_flow = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available() and  state.args.gpu:
            data = data.to(device_id)
            target = target.to(device_id)  
        state.optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = state.criterior(output, target)
        running_loss += loss.item()
        i = i + 1


        loss.backward()
        if state.grad_flow_enabler :
            grad_flow = calc_grad_flow(model.module.named_parameters())
        state.optimizer.step()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        total_checked += data.size(0)

    acc = 100. * correct / total
    running_loss /= len(train_loader)
    return running_loss, acc, (total_checked,grad_flow)

def Test_Model(state):

    model = state.model
    device_id = state.device_id
    criterior = state.criterior
    if state.test_final :
        test_loader = state.data_pack["final_test_loader"]
    else:
        test_loader = state.data_pack["test_loader"]
    acc1 = utility.AverageMeter("Acc@1", ":6.2f", utility.Summary.AVERAGE)
    acc5 = utility.AverageMeter("Acc@5", ":6.2f", utility.Summary.AVERAGE)
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        correct_prediction = 0.0
        total_pred = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available() and state.args.gpu:
                data = data.to(device_id)
                target = target.to(device_id)  
            output = model(data)
            
            batch_size = data.size(0)
            top1, top5 = utility.accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0].item(), batch_size)
            acc5.update(top5[0].item(), batch_size)

            loss = criterior(output, target).detach()
            running_loss += loss.item()

    running_loss /= len(test_loader)
    acc = (acc1.avg, acc5.avg)

    return running_loss, acc


def calc_grad_flow(named_parameters):
    layers = []
    k = 0
    res_mean = []
    res_std = []
    max_grads = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and \
                ("p1" not in n) and ("p2" not in n) and ("p3" not in n) and ("p4" not in n) and ("p5" not in n):
            layers.append(n)
            res_mean.append(p.grad.abs().mean().cpu())
            res_std.append(p.grad.abs().std().cpu())
            max_grads.append(p.grad.abs().max().cpu())
            k += 1
    res_mean = np.array(res_mean)
    res_std = np.array(res_std)
    max_grads = np.array(max_grads)
    return (res_mean, res_std, max_grads)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and \
            ("p1" not in n) and ("p2" not in n) and ("p3" not in n) and ("p4" not in n) and ("p5" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
