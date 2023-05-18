import os
import torch
import copy
import logging
import settings 
import utility 
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import training.train_lib as train_lib
import optimization.smac_opt as smac_opt
from optimization.optuna_lib import HPO_optuna
import optimization.optimization_lib as AF_optim
import optimization.cross_validation as cross_val
from torch.nn.parallel import DistributedDataParallel as DDP
utility.set_logging()


def fixed_activation(state):
    utility.Set_Seed(state.args)
    train_dense_model(state)
    test_loss, test_acc = train_lib.Test_Model(state)
    logging.info("pruned model befor train_with_selected_activation: Test Accuracy=%s, Test Loss =%s"%(test_acc, test_loss))
    train_with_selected_activation(state, "gelu")#TODO:

def train_with_selected_activation(state, activation_info):
    args = state.args
    if args.d != 2:
        state.kfold_enabler = True
    else:
        state.kfold_enabler = False
    activation = activation_info

    checkpoint_name = "prune%s.pth"%(state.MODEL_ARCHITECTURES[args.m])
    state.Load_Checkpoint(checkpoint_name)

    for i,item in enumerate(state.model.optim_dictionary):
        state.model.optim_dictionary[item][2][0] = activation

    logging.info("train_with_selected_activation: %s" %state.model.optim_dictionary)
    logging.info('-' * 20)

    state.requires_grad_enabler = False
    utility.set_requires_grad(state)    
    utility.Show_Gradients(state.model)

    if state.kfold_enabler:
        mode = "train_selected_activation_%s" % (activation)
        cross_val.kfold_training(state, mode)
    else:
        mode = "train_selected_activation_%s" % (activation)
        training(state, "No_Save")   

    test_loss, test_acc = train_lib.Test_Model(state)
    logging.info("model_with_%s: Test Accuracy=%s, Test Loss =%s"\
                %(activation, test_acc, test_loss))

def run_fig6(state):
    utility.Set_Seed(state.args)
    train_dense_model(state)
    test_loss, test_acc = train_lib.Test_Model(state)
    logging.info("pruned model: Test Accuracy=%s, Test Loss =%s"%(test_acc, test_loss))
    second_stage(state)


def metafire (state):
    #utility.Set_Seed(state.args)
    test_loss, test_acc = train_lib.Test_Model(state)

    train_dense_model(state)
    test_loss, test_acc = train_lib.Test_Model(state)
    logging.info("pruned model: Test Accuracy=%s, Test Loss =%s"%(test_acc, test_loss))
    first_stage(state)
    second_stage(state)

def first_stage(state):
    utility.Set_Seed(state.args)
    #grad_disable
    state.requires_grad_enabler = False
    utility.set_requires_grad(state)
    args = state.args
    state.kfold_enabler = 0
    mode = "retrain_after_search_%s_PR_%s"%(state.MODEL_ARCHITECTURES[args.m],args.pruning_rate)
    if args.first_stage :
        logging.info("First_stage: Searching AF")
        args.training_enabler = False
        state = AF_optim.AF_Operation_Optimization(state)
        args.training_enabler = True
        logging.info("First_stage: output optim_dictionary:") 
        logging.info(state.show_optim_dictionary()) 
        logging.info("First_stage: Retrain_after_search")

        if state.args.d != 2 and state.args.m!=1:
            state.kfold_enabler = 1
        state.grad_flow_enabler = True
        activation =  ['symexp', 'relu6', 'hardswish', 'acon']
        for i,item in enumerate(state.model.optim_dictionary):
            state.model.optim_dictionary[item][2][0] = activation[i]
        if state.kfold_enabler:
            mode = 'aftersearch_ab'
            cross_val.kfold_training(state, mode)
        else:
            training(state, mode) 
            
        state.grad_flow_enabler = False

        test_loss, test_acc = train_lib.Test_Model(state)
        logging.info("Prune and Retrain_after_search: Test Accuracy:%s , Test Loss =%s" %(test_acc, test_loss))
    else:
        pass
        # with open(state.CHECKPOINT_PATH+"K_Fold/best_%s.txt"%mode, "r") as f:
        #     best_name = f.read()
        # state.Load_Checkpoint("K_Fold/"+best_name) 

    ch_name = "prune%s.pth"%(state.MODEL_ARCHITECTURES[state.args.m])
    state.Load_Checkpoint(ch_name)
    test_loss, test_acc = train_lib.Test_Model(state)
    logging.info("model first stage: Test Accuracy=%s, Test Loss =%s  Activation Function:%s"\
                %(test_acc, test_loss, state.show_optim_dictionary()))
    utility.Show_Gradients(state.model)
    print(state.model.optim_dictionary)
    logging.info('-' * 30)

def Load_Checkpoint(state, checkpoint_name):
    # Load the training model

    if 'prune' in checkpoint_name: #TODO:
        checkpoint = torch.load(state.CHECKPOINT_PATH + checkpoint_name)
    elif 'first_train' in checkpoint_name:
        checkpoint = torch.load(state.CHECKPOINT_dense_PATH + checkpoint_name)
        for item in state.model.optim_dictionary.keys():
            state.model.optim_dictionary[item][2][0] = checkpoint['optim_dictionary'][item][2][0]
        state.kfold_indexes_dense =  checkpoint['kfold_indexes']
    else :
        checkpoint = torch.load(checkpoint_name,map_location="cpu")
        for item in state.model.optim_dictionary.keys():
            state.model.optim_dictionary[item][2][0] = checkpoint['optim_dictionary'][item][2][0]
        
    state.model.load_state_dict(checkpoint['model_state_dict'])
    #state.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    state.epoch = checkpoint['epoch']
    state.kfold_indexes =  checkpoint['kfold_indexes']

def second_stage(state):
    utility.Set_Seed(state.args)
    args = state.args
    args.training_enabler = False
    state.requires_grad_enabler = True
    utility.set_requires_grad(state)
    logging.info("Start Second Stage")
    utility.Show_Gradients(state.model)
    smac_opt.run_strategies.Run_Smac(state)



        

def training(state, mode):
    #utility.Set_Seed(state.args)

    ctx = mp.get_context('spawn')
    p2c = ctx.SimpleQueue()

    utility.set_requires_grad(state)
    mp.spawn(worker, nprocs=state.args.gpus, args=(state, mode, p2c))
    for _ in range(state.args.gpus):
        dev , acc =p2c.get()
    state.device_id = state.args.set_device
    torch.cuda.set_device(state.device_id)
    return acc 


def worker(device_id, state, mode, result_queue):
    #utility.Set_Seed(state.args)

    args = state.args
    if args.gpus == 1:
        rank_id = args.nr * args.gpus + device_id
        device_id = args.set_device
    else :
        rank_id = args.nr * args.gpus + device_id
    state.device_id = device_id
    state.rank_id = rank_id

        
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank_id
    )
    torch.cuda.set_device(device_id)
    
    #  Use new kfold indexes for each fold training and use dense saved indexes for 1 fold training
    if args.d != 2 and state.args.m!=1: #IMAGNET has train val and test subset and kfold is not enabled for it.
        if state.kfold_enabler:
            cross_val.setup_dataflow(state, state.kfold_indexes)
        else:
            cross_val.setup_dataflow(state, state.kfold_indexes_dense)
    # prepare model, update_optimizer and criterior
    state.model.to(device_id)
    state.model = DDP(state.model, device_ids=[device_id],find_unused_parameters=True)
    state.update_optimizer()
    state.criterior = torch.nn.CrossEntropyLoss().to(device_id)
    train_lib.Training(state, mode, result_queue)
    cleanup()
    
def cleanup():
    dist.destroy_process_group()


def train_dense_model(state):
    utility.Set_Seed(state.args)
    args = state.args
    if args.d==2 or state.args.m==1:
        state.kfold_enabler = 0
    else:
        state.kfold_enabler = 1 
    state.requires_grad_enabler = False
    utility.set_requires_grad(state)
    mode = "first_train_%s"%(state.MODEL_ARCHITECTURES[args.m])
    if  args.first_train:
        if state.args.d != 2 and state.args.m!=1:
            state.save_dataset()
        if state.kfold_enabler:
            cross_val.kfold_training(state, mode)
        else:
            training(state, mode)   
        
        test_loss, test_acc = train_lib.Test_Model(state)
        logging.info("Dense Model: Test Accuracy=%s, Test Loss =%s"%(test_acc, test_loss))
        logging.info('-' * 30)
    else :
        if state.args.d != 2 and state.args.m!=1:
            state.load_dataset()
        if state.kfold_enabler:
            with open(state.CHECKPOINT_dense_PATH+"K_Fold/best_%s.txt"%mode, "r") as f:
                check_name = f.read()
                    
                
                state.Load_Checkpoint("K_Fold/"+check_name) 
        else:
            ch_name = "%s.pth"%(mode)
            state.Load_Checkpoint(ch_name)  

    state.kfold_enabler = 0  
     
    if state.args.d != 2 and state.args.m!=1:
        state.kfold_enabler = 1  
        cross_val.setup_dataflow(state, state.kfold_indexes_dense)  
      
    test_loss, test_acc = train_lib.Test_Model(state)
    logging.info("Dense Model: Test Accuracy=%s, Test Loss =%s"%(test_acc, test_loss))
        
    utility.Prepare_Model_secound_method(state)
    
    test_loss, test_acc = train_lib.Test_Model(state)
    logging.info("pruned model: Test Accuracy=%s, Test Loss =%s"%(test_acc, test_loss))
    #TODO: train after prune
    if state.args.d != 2 and state.args.m!=1:
        state.kfold_enabler = 1
    mode = "Train_after_prune_%s_PR_%s"%(state.MODEL_ARCHITECTURES[args.m], args.pruning_rate)
    if  args.Train_after_prune:  
        state.grad_flow_enabler = True
        if state.kfold_enabler:
            cross_val.kfold_training(state, mode)
        else:
            training(state, mode)   
        state.grad_flow_enabler = False
  
        test_loss, test_acc = train_lib.Test_Model(state)
        logging.info("Train_after_prune: Test Accuracy=%s, Test Loss =%s"%(test_acc, test_loss))
        logging.info('-' * 30)
