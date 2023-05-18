import os
import copy
import torch
import shutil
import numpy as np
import utility as utility
import settings as settings
import torch.multiprocessing as mp
from sklearn.model_selection import KFold
import pruning.pruning_lib as pruning_lib
import torchvision.transforms as transforms
import logging 
utility.set_logging()

def setup_dataflow(state, kfold_indexes):
    #utility.Set_Seed(state.args)
    args = state.args
    rank_id = state.rank_id
    train_idx, valid_idx = kfold_indexes
    dataset = state.data_pack["kfold_dataset"]
    train = torch.utils.data.dataset.Subset(dataset, train_idx)
    valid = torch.utils.data.dataset.Subset(dataset, valid_idx)

    if state.args.gpus>1:
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
        train, num_workers = args.workers,
        batch_size=args.b, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        valid, num_workers = args.workers,
        batch_size=args.b, shuffle=False,
         pin_memory=True)
    state.data_pack["train_loader"] = train_loader
    state.data_pack["test_loader"] = val_loader

def kfold_training(state, mode):
    args = state.args
    # define kfold
    splits = KFold(n_splits=args.num_folds, shuffle=True, random_state=settings.SEED)
    dataset = state.data_pack["kfold_dataset"]
    first_state_dict = copy.deepcopy(state.model.state_dict())
    temp_result =[]
    for fold_idx, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        state.kfold_indexes = (train_idx, val_idx)
        logging.info('Fold {}'.format(fold_idx + 1))
        ctx = mp.get_context('spawn')
        p2c = ctx.SimpleQueue()
        kfold_mode = "K_Fold/Fold%s_%s" %(fold_idx, mode)
        mp.spawn(pruning_lib.worker, nprocs=args.gpus, args=(state, kfold_mode, p2c))
        state.device_id = state.args.set_device
        torch.cuda.set_device(state.device_id)
        state.model.load_state_dict(first_state_dict)  #TODO:select beast model weghts
        
        for _ in range(state.args.gpus):
            device, test_acc = p2c.get()
            temp_result.append((test_acc, device, fold_idx))
    if  "first_train" in mode : 
        test_acc, device, fold_idx = max(temp_result)
        best_name = "Fold%s_%s.pth" %(fold_idx, mode)  
        best_dir = state.CHECKPOINT_dense_PATH+"K_Fold/"+best_name
        with open(state.CHECKPOINT_dense_PATH+"K_Fold/best_%s.txt"%mode, "w") as f:
            f.write(best_name)
        state.kfold_indexes_dense = (train_idx, val_idx)#TODO:   
    else:
        test_acc, device, fold_idx = max(temp_result)
        best_name = "Fold%s_%s.pth" %(fold_idx, mode)  
        best_dir = state.CHECKPOINT_PATH+"K_Fold/"+best_name
        with open(state.CHECKPOINT_PATH+"K_Fold/best_%s.txt"%mode, "w") as f:
            f.write(best_name)   
    
    return test_acc
    
    

