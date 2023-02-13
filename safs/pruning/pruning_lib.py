import torch
import logging
import settings 
import utility 
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import training.train_lib as train_lib
import optimization.hpo_lib as hpo_lib
from optimization.optuna_lib import HPO_optuna
import optimization.optimization_lib as AF_optim
import optimization.cross_validation as cross_val
from torch.nn.parallel import DistributedDataParallel as DDP



device = 'cuda' if torch.cuda.is_available() else 'cpu'
def optuna_runing(PR_ratio, model, args, criterior, optimizer, train_loader, test_loader):
        utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)
        for name, param in model.named_parameters():
            param.requires_grad = True
            if 'alpha' in name or 'beta' in name or 'acon_pp' in name:
                    param.requires_grad = False

        if settings.First_train:
            model = train_lib.Training('first_train', model, criterior, optimizer, train_loader)
            test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
            logging.info('Original Model: Test Accuracy, Test Loss = %f , %f', test_acc, test_loss)

            logging.info('-' * 20)
        else :
            model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer, 'Model_first_train' +
                                                            settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')
            test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
            logging.info('Original Model: Test Accuracy, Test Loss = %f , %f', test_acc, test_loss)
        utility.Prepare_Model_secound_method(model, PR_ratio, args, training_mode="Prune")  # TODO setting

        checkpoint = utility.Create_Checkpoint(0, model.state_dict(), optimizer.state_dict(), test_acc, model.optim_dictionary, test_loss)
        utility.Save_Checkpoint(checkpoint,
                                'Model_prune' + settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')
        test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
        logging.info('prune fine: Test Accuracy, Test Loss = %f , %f', test_acc, test_loss)
        settings.Model = model
        HPO_optuna()

def pruning_test_kfold (PR_ratio, model, args, criterior, optimizer, train_loader, test_loader):
        utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)
        for name, param in model.named_parameters():
            param.requires_grad = True
            if 'alpha' in name or 'beta' in name or 'acon_pp' in name:
                    param.requires_grad = False

        if settings.First_train:
            model = train_lib.Training('first_train', model, criterior, optimizer, train_loader)
            test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
            logging.info('Original Model: Test Accuracy, Test Loss = %f , %f', test_acc, test_loss)

            logging.info('-' * 20)
        else :
            model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer, 'Model_first_train' +
                                                            settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')
            test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
            logging.info('Original Model: Test Accuracy, Test Loss = %f , %f', test_acc, test_loss)


        utility.Prepare_Model_secound_method(model, PR_ratio, args, training_mode="Prune")  # TODO setting

        checkpoint = utility.Create_Checkpoint(0, model.state_dict(), optimizer.state_dict(), test_acc, model.optim_dictionary, test_loss)
        utility.Save_Checkpoint(checkpoint,
                                'Model_prune' + settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')

        test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
        logging.info('prune fine: Test Accuracy, Test Loss = %f , %f', test_acc, test_loss)

        train_with_selected_activation_kfold(model, settings.run_activation, \
            settings.run_activation_ab_flag, train_loader, test_loader, criterior)

        return model    


def pruning_test (PR_ratio, model, args, criterior, optimizer, train_loader, test_loader):

        if settings.First_train:
            model = train_lib.Training('first_train', model, criterior, optimizer, train_loader)
            test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
            logging.info('Original Model: Test Accuracy, Test Loss =', str(test_acc), str(test_loss))
            logging.info('-' * 20)
        else :
            model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer, 'Model_first_train' +
                                                            settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')
            test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
            logging.info('Original Model: Test Accuracy, Test Loss =', str(test_acc), str(test_loss))


        utility.Prepare_Model_secound_method(model, PR_ratio, args, training_mode="Prune")  # TODO setting

        checkpoint = utility.Create_Checkpoint(0, model.state_dict(), optimizer.state_dict(), test_acc, model.optim_dictionary, test_loss)
        utility.Save_Checkpoint(checkpoint,
                                'Model_prune' + settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')

        test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
        logging.info('prune fine: Test Accuracy, Test Loss =', test_acc, test_loss)

        train_with_selected_activation(model, settings.run_activation, \
            settings.run_activation_ab_flag, train_loader, test_loader, criterior)

        return model


def metafire (PR_ratio, model, args, criterior, optimizer, train_loader, test_loader):

    utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)

    for name, param in model.named_parameters():
        param.requires_grad = True
        if 'alpha' in name or 'beta' in name or 'acon_pp' in name:
                param.requires_grad = False

    if settings.First_train:
        model = train_lib.Training('first_train', model, criterior, optimizer, train_loader)
        test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
        logging.info('Original Model: Test Accuracy, Test Loss = %f , %f', test_acc, test_loss)
        logging.info('-' * 20)
    else :
        model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer, 'Model_first_train' +
                                                        settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')
        test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
        logging.info('Original Model: Test Accuracy, Test Loss = %f , %f', test_acc, test_loss)

    utility.Prepare_Model_secound_method(model, PR_ratio, args, training_mode="Prune")  # TODO setting
    checkpoint = utility.Create_Checkpoint(0, model.state_dict(), optimizer.state_dict(), test_acc, model.optim_dictionary, test_loss)
    utility.Save_Checkpoint(checkpoint,
                            'Model_prune' + settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')

    if settings.Train_after_prune:
        logging.info('Train_after_prune...')
        optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                              momentum=settings.Optim_default['momentum'],
                              weight_decay=settings.Optim_default['wd'])
        utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)
        model = train_lib.Training('train_after_prune', model, criterior, optimizer, train_loader)
        test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
        logging.info("Prune and retrain with relu: Test Accuracy:%s , Test Loss =%s" %(test_acc, test_loss))
    else:
        model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer, 'Model_train_after_prune' +
                                                         settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')
        test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
        logging.info("Prune and retrain with relu: Test Accuracy:%s , Test Loss =%s" %(test_acc, test_loss))


    model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer, 'Model_prune' +
                                                     settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')

    test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
    logging.info("ready to search-Prune with relu: Test Accuracy:%s , Test Loss =%s" %(test_acc, test_loss))
    logging.info('-' * 20)

    ##################################     search AF      ##################################
    for name, param in model.named_parameters():
        param.requires_grad = True
        if 'alpha' in name or 'beta' in name or 'acon_pp' in name:
            param.requires_grad = False

    if settings.First_stage :
        optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                              momentum=settings.Optim_default['momentum'],
                              weight_decay=settings.Optim_default['wd'])
        utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)
        
        logging.info("First_stage : Searching AF")
        settings.TRAINING_ENABLER = False
        model = AF_optim.AF_Operation_Optimization(model, criterior, optimizer, train_loader)
        settings.TRAINING_ENABLER = True

        logging.info(str(model.optim_dictionary)) #TODO:
        
        logging.info("First_stage : Retrain_after_search")
        optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                              momentum=settings.Optim_default['momentum'],
                              weight_decay=settings.Optim_default['wd'])
        utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)    
        model = train_lib.Training('Retrain_after_search', model, criterior, optimizer, train_loader)
        test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
        logging.info("Prune and Retrain_after_search: Test Accuracy:%s , Test Loss =%s" %(test_acc, test_loss))


        # load prune weights
        model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer,
                                                                   'Model_prune' + settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')
    else:
        model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer,
                                                 'Model_Retrain_after_search' + settings.MODEL_ARCHITECTURES[
                                                    settings.MODEL_ARCHITECTURE] + '.pth')
        model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer,
                                                         'Model_prune' + settings.MODEL_ARCHITECTURES[
                                                             settings.MODEL_ARCHITECTURE] + '.pth')

    ############################## Fine Tune - Alpha and Beta ###############################
    test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
    logging.info("Prune with searched AF model: Test Accuracy:%s , Test Loss =%s" %(test_acc, test_loss))
    logging.info(str(model.optim_dictionary)) 

    logging.info("Second stage: Activate alpha and beta")
    for name, param in model.named_parameters():
        param.requires_grad = True
        if 'alpha' in name or 'beta' in name or 'acon_pp' in name:
            param.requires_grad = True

    logging.info("Second stage: retrain searched model with alpha and beta params")
    optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                            momentum=settings.Optim_default['momentum'],
                            weight_decay=settings.Optim_default['wd'])

    utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)
    model = train_lib.Training('Train_ab_after_search', model, criterior, optimizer, train_loader)


    ##################################     HPO           ##################################
    logging.info("Second stage: HPO")
    model, optimizer, _, _ = utility.Load_Checkpoint(model, optimizer,
                                                    'Model_prune' + settings.MODEL_ARCHITECTURES[
                                                        settings.MODEL_ARCHITECTURE] + '.pth')
    settings.Model = model
    logging.info(str(model.optim_dictionary)) 
    HPO_optuna()

    return model

def train_with_selected_activation(model, args, activation, a_b_flag, train_loader,test_loader, criterior):

    model, _, _, _ = utility.Load_Checkpoint(model, settings.Optimizer,
                                                     'Model_prune' + settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')
 
    for i,item in enumerate(model.optim_dictionary):
        model.optim_dictionary[item][2][0] = activation

    logging.info("train_with_selected_activation"+str(model.optim_dictionary))
    logging.info('-' * 20)

    for name, param in model.named_parameters():
        param.requires_grad = True
        if 'alpha' in name or 'beta' in name or 'acon_pp' in name:
            if a_b_flag:
                param.requires_grad = True
            else:
                param.requires_grad = False



    mode = "train_selected_activation_%s_%s" % (a_b_flag, activation)
    mp.spawn(worker, nprocs=args.gpus, args=(args,model,mode))
    
    test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
    logging.info('prune fine tune search: Test Accuracy, Test Loss =', test_acc, test_loss)
    return model


def worker(device_id, args):
    model = args[1]
    mode = args[2]
    args = args[0]
    
    rank_id = args.nr * args.gpus + device_id
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank_id
    )
    torch.cuda.set_device(device_id)
    if settings.DATASET == 0:
        train_loader, test_loader = utility.MNIST_Loaders(settings.BATCH_SIZE, settings.WORKERS)
    else:
        all_dataset, train_loader, test_loader, final_test_loader =\
             utility.CIFAR10_Loaders(args, rank_id, settings.BATCH_SIZE,\
                 settings.WORKERS)
    ##################### store global setting and variable #####################
    settings.DATA_Test = test_loader
    settings.DATA_Train = train_loader
    settings.All_Dataset = all_dataset
    settings.DATA_Final_Test = final_test_loader

    optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                            momentum=settings.Optim_default['momentum'],
                            weight_decay=settings.Optim_default['wd'])
    utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)

    model = DDP(model, device_ids=[device_id])
    model = train_lib.Training(model , mode, device_id, args, settings.Criterior,\
         optimizer, train_loader)




def train_with_selected_activation_kfold(model, activation, a_b_flag, train_loader,test_loader, criterior):

    model, _, _, _ = utility.Load_Checkpoint(model, settings.Optimizer,
                                                     'Model_prune' + settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')
 
    for i,item in enumerate(model.optim_dictionary):
        model.optim_dictionary[item][2][0] = activation

    # model.apply(init_weights)
    logging.info("train_with_selected_activation"+str(model.optim_dictionary))
    # prune fine tune
    logging.info('-' * 20)
    if settings.Finetune_a_b :
        # settings.TRAIN_EPOCH = 30
        for name, param in model.named_parameters():
            param.requires_grad = True
            if 'alpha' in name or 'beta' in name or 'acon_pp' in name:
                if a_b_flag:
                    param.requires_grad = True
                else:
                    param.requires_grad = False



        optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                              momentum=settings.Optim_default['momentum'],
                              weight_decay=settings.Optim_default['wd'])
        utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)
        model, _, _  = cross_val.Kfold_Training("train_selected_activation_"+str(a_b_flag)+"_"+activation, model, 3, criterior, optimizer)
        test_loss, test_acc = train_lib.Test_Model(model, criterior, test_loader)
        logging.info('prune fine tune search: Test Accuracy, Test Loss =', test_acc, test_loss)
    return model 

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

