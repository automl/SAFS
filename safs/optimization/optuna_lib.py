import utility
import logging
import optuna
import settings as settings
import torch.optim as optim
import training.train_lib as train_lib
from tmomentum.optimizers import TAdam
from optimization.fromage import Fromage


def objective(trial):
    model = settings.Model
    model, _, _, _ = utility.Load_Checkpoint(model, settings.Optimizer,
                                                     'Model_prune' + settings.MODEL_ARCHITECTURES[
                                                         settings.MODEL_ARCHITECTURE] + '.pth')

    optim_name = trial.suggest_categorical("optim_name", ["SGD", "TAdam","Fromage","Adam"])#,"SparseAdam"

    logging.info('optim_name= %s' % (optim_name))
    lr_scheduler_ = "first_run"
    if optim_name == "SGD":
        lr_scheduler = trial.suggest_categorical("lr_scheduler",
                                            ["Constant", "Cosine", "Step", "Linear", "ReduceLROnPlateau"])
        lr_scheduler_ = lr_scheduler
        momentum = trial.suggest_float("momentum", 0.1, 0.99)
        # weight_decay = trial.suggest_float("weight_decay", 0, 100)
        settings.Optim_default['momentum'] = momentum
        # settings.Optim_default['wd'] = weight_decay
        optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                              momentum=settings.Optim_default['momentum'],
                              weight_decay=settings.Optim_default['wd'])

        utility.Set_Lr_Policy(lr_scheduler, optimizer)
        logging.info('SGD : momentum= %s  lr_scheduler=%s' % (momentum,lr_scheduler))

        # getattr(optim, optim_name)(lr)
    elif optim_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    elif optim_name == "TAdam":
        optimizer = TAdam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    elif optim_name == "SparseAdam":
        optimizer = optim.SparseAdam(model.parameters(),lr=0.001, betas=(0.9, 0.99))
    elif optim_name == "Fromage":
        optimizer = Fromage(model.parameters(), lr=0.01)

    criterior = settings.Criterior
    if optim_name == "SGD":
        model = train_lib.Training('model_%s_m%s_%s' %(optim_name,settings.Optim_default['momentum'],lr_scheduler_), model, criterior, optimizer, settings.DATA_Train)
    else:
        model = train_lib.Training('model_%s' %(optim_name), model, criterior, optimizer, settings.DATA_Train)
    test_loss, test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Test)
    print('HPO: Test Accuracy, Test Loss =', test_acc, test_loss)
    logging.info('HPO: Test Accuracy, Test Loss = %s %s' % (test_acc, test_loss))
    return float(test_acc / 100)



def  HPO_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    # study.optimize(objective, timeout=2.0)
    print(study.best_trial)
    logging.info("best_result: %s" %(study.best_trial))



