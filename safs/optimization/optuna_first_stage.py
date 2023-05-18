import utility
import logging
import os
import optuna
import settings as settings
import torch.optim as optim
import training.train_lib as train_lib
from tmomentum.optimizers import TAdam
from optimization.fromage import Fromage
import pruning.pruning_lib as pruning_lib






class HPO_Model:
    def __init__(self, state):
        self.state = state
         
    def objective(self, trial):

        AF1 = trial.suggest_categorical("AF1", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF2 = trial.suggest_categorical("AF2", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF3 = trial.suggest_categorical("AF3", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF4 = trial.suggest_categorical("AF4", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF5 = trial.suggest_categorical("AF5", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF6 = trial.suggest_categorical("AF6", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF7 = trial.suggest_categorical("AF7", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF8 = trial.suggest_categorical("AF8", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF9 = trial.suggest_categorical("AF9", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF10 = trial.suggest_categorical("AF10", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF11 = trial.suggest_categorical("AF11", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF12 = trial.suggest_categorical("AF12", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF13 = trial.suggest_categorical("AF13", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF14 = trial.suggest_categorical("AF14", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        AF15 = trial.suggest_categorical("AF15", ['relu6','hardswish','swish','TanhSoft-1',\
                'acon', 'gelu','srs', 'elu','tanh', 'softplus','logsigmiod','symlog','symexp'])
        active = [AF1, AF2, AF3, AF4, AF5, AF6, AF7, AF8, AF9, AF10, AF11, AF12, AF13, AF14,AF15]
        print(active)
        for i in self.state.model.optim_dictionary:
            self.state.model.optim_dictionary[i][2][0] = active[counter]
            counter += 1
    
        self.state.requires_grad_enabler = False
        utility.set_requires_grad(self.state)
        file_list = os.listdir(self.state.CHECKPOINT_PATH)
        for i in range(30):
            mode = "search_%s"%(i)
            if not( mode+'.pth' in file_list):
                break
        test_acc = pruning_lib.training(self.state, mode)              
        logging.info(self.input_state.show_optim_dictionary()) #TODO:
        logging.info("test_acc: %s " %test_acc)
        logging.info("")
        with open("%ssearched_activations_%s.txt"%\
            (self.input_state.CHECKPOINT_PATH,self.input_state.args.o), "a") as f:
            f.write("%s acc:%s \n "%(self.state, test_acc))
            f.close()

        print('HPO: Test Accuracy =', test_acc)
        return float(test_acc / 100)
        




def  HPO_optuna(state):
    study = optuna.create_study(direction="maximize")
    model = HPO_Model(state)
    study.optimize(model.objective, n_trials=20)
    # study.optimize(objective, timeout=2.0)
    print(study.best_trial)
    logging.info("best_result: %s" %(study.best_trial))



