import numpy as np
import time
import training.train_lib as train_lib
from sklearn.model_selection import KFold
import utility as utility
import settings as settings
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset


def setup_dataflow(dataset, train_idx, val_idx):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE, sampler=val_sampler)
    return train_loader, val_loader


def save_kfold_model(model, model_name, fold_idx, results_per_epoch):
    if model_name == "No_Save":
        pass
    else :
        checkpoint = utility.Create_Checkpoint(fold_idx, model.state_dict(),settings.Optimizer.state_dict(), 1,model.optim_dictionary,results_per_epoch)
        utility.Save_Checkpoint(checkpoint,
                                "K_Fold/KFold_%s%s%s.pth" %(fold_idx, model_name ,settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE]))


def Kfold_Training(model_name, model, k_folds, criterior, optimizer):
    if settings.TRAINING_ENABLER == True:
        epoch = settings.TRAIN_EPOCH
    else:
        epoch = settings.OPTIM_EPOCH
    # define kfold
    splits = KFold(n_splits=k_folds, shuffle=True)
    dataset = settings.All_Dataset
    results_per_fold = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        model, _, _, _ = utility.Load_Checkpoint(model, settings.Optimizer,
                                                 'Model_prune' + settings.MODEL_ARCHITECTURES[
                                                     settings.MODEL_ARCHITECTURE] + '.pth')
        print('Fold {}'.format(fold_idx + 1))
        train_loader, val_loader = setup_dataflow(dataset, train_idx, val_idx)

        utility.Set_Lr_Policy(settings.lr_scheduler_name, optimizer)
        scheduler = settings.LR_SCHEDULER
        
        logging = settings.LOGGING
        results_per_epoch = []

        for i in range(epoch):
            start_time = time.time()
            train_loss, acc, total_checked = train_lib.Train_Model(model, criterior, optimizer, train_loader)
            test_loss1, test_acc1 = train_lib.Test_Model(model, criterior, val_loader)
            test_loss2, test_acc2 = train_lib.Test_Model(model, criterior, settings.DATA_Test)

            lr = scheduler.get_last_lr()[0]
            # logging.info('epoch %d lr %e', epoch, lr)
            if settings.lr_scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(train_loss)          
            else:
                scheduler.step()



            mean_, std_ = train_lib.calc_grad_flow(model.named_parameters())
            total_grad_mean = np.zeros(len(mean_))
            total_grad_std = np.zeros(len(std_))
            total_grad_mean = [gbase1 + g1 * float(total_checked) for gbase1, g1 in zip(total_grad_mean, mean_)]
            total_grad_std = [gbase1 + g1 * float(total_checked) for gbase1, g1 in zip(total_grad_std, std_)]
            total_grad_mean = [gbase1 / total_checked for gbase1 in total_grad_mean]
            total_grad_std = [gbase1 / total_checked for gbase1 in total_grad_std]
            
            end_time = time.time()
            epoch_time = end_time - start_time

            logging.info("epoch_{} time:{} lr:{} train_acc:{} test_acc:{}".format(i, epoch_time, lr, acc, test_acc1))

            

            results_per_epoch.append({"train_acc":(train_loss, acc), "test_acc1":(test_loss1, test_acc1),
                                      "test_acc2":(test_loss2, test_acc2),"grad_flow":(total_grad_mean, total_grad_std),
                                      "epoch_time":epoch_time})

        results_per_fold.append(results_per_epoch)
        save_kfold_model(model, model_name, fold_idx, {'results_per_epoch':results_per_epoch})
    # sum_up_result, result_per_fold = result_fold(results_per_fold)
    # print('sum_up_result=tarin,val_per_train,test:(loss,acc)')
    # print(sum_up_result)
    # print(result_per_fold)
    return model, 0, 0


def result_fold(results_per_fold):
    # sum_up_result  totally [(train_loss, acc), (test_loss1, test_acc1), (test_loss2, test_acc2)]
    # result_per_fold   per fold result
    results_per_fold = np.array(results_per_fold)
    total = []
    for item in results_per_fold:
        for i in range(3):
            total.append((np.mean(item[:, i], axis=0)))
    total = np.array(total)
    result_per_fold = total.reshape((len(results_per_fold), 3, 2))
    sum_up_result = np.mean(total.reshape((len(results_per_fold), 3, 2)), axis=0)

    return sum_up_result, result_per_fold


def fold_result_analise(results_per_fold, num_folds):
    acc_sum = 0
    for n_fold in range(len(results_per_fold)):
        current_fold = results_per_fold[n_fold]
        print(
            f"Validation Results - Fold[{n_fold + 1}] Avg accuracy: {current_fold[1][2]['Accuracy']:.2f} Avg loss: {current_fold[1][2]['Loss']:.2f}")
        acc_sum += current_fold[1][2]['Accuracy']

    folds_mean = acc_sum / num_folds
    print(f"Model validation average for {num_folds}-folds: {folds_mean :.2f}")

