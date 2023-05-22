
import logging
import utility
import numpy as np
import torch.nn as nn
import settings as settings
import pruning.pruning_lib as pruning_lib
import torch.optim as optim
from utility import Set_Lr_Policy
import training.train_lib as train_lib
from tmomentum.optimizers import TAdam
from optimization.fromage import Fromage
from smac.scenario.scenario import Scenario
from ConfigSpace.conditions import InCondition
import optimization.cross_validation as cross_val
from ConfigSpace import Configuration, ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.configspace import CategoricalHyperparameter,\
                             UniformFloatHyperparameter,\
                             UniformIntegerHyperparameter,\
                             Constant
utility.set_logging()



class run_strategies:
    @staticmethod
    def Run_Smac(state):
        utility.Set_Seed(state.args)

        state.kfold_enabler = False
        state.args.training_enabler = False
        model = HPO_Model(state)
        scenario = Scenario(
                {   
                    "run_obj": "quality",
                    "runcount-limit": state.args.hpo_steps,
                    "cs": model.configspace,
                    "output-dir" : state.CHECKPOINT_PATH+"SMAC_output_seed%s/"%state.args.random_seed
                }           )

        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(state.args.random_seed),
                        tae_runner=model.train
                        )

        
        best_found_config = smac.optimize()
        tae = smac.get_tae_runner()
        
        if state.args.d != 2:
            state.kfold_enabler = 1
        state.grad_flow_enabler = True
        state.args.training_enabler = True
        best_found_cost = (1-tae.run(config=best_found_config)[1])*100
        logging.info("best_result: %s acc:%s" %(best_found_config, best_found_cost))#TODO:




class HPO_Model:
    def __init__(self, state):
        self.state = state

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        optim_name = CategoricalHyperparameter("optim_name", ['SGD', 'TAdam','Fromage','Adam'], 'SGD')
        cs.add_hyperparameter(optim_name)

        lr_scheduler = CategoricalHyperparameter('lr_scheduler',\
            ['VGG_lr_scheduler','CosineWR' , 'Constant', 'Cosine', 'Step', 'Linear', 'ReduceLROnPlateau'],\
                default_value=self.state.lr_scheduler_name)
        cs.add_hyperparameter(lr_scheduler)

        momentum = UniformFloatHyperparameter('momentum', 0.1, 0.999, default_value=0.9)
        cs.add_hyperparameter(momentum)

        lr_SGD = UniformFloatHyperparameter('lr_SGD', 1e-4, 1e-1, log=True, default_value=self.state.Optim_default_params["learning_rate"])
        cs.add_hyperparameter(lr_SGD)

        lr_Adam = UniformFloatHyperparameter('lr_Adam', 1e-4, 1e-1, log=True, default_value=1e-3)
        cs.add_hyperparameter(lr_Adam)

        lr_Fromage = UniformFloatHyperparameter('lr_Fromage', 1e-4, 1e-1, log=True, default_value=1e-3)
        cs.add_hyperparameter(lr_Fromage)

        lr_TAdam = UniformFloatHyperparameter('lr_TAdam', 1e-4, 1e-1, log=True, default_value=1e-3)
        cs.add_hyperparameter(lr_TAdam)
    
        wd_TAdam = UniformFloatHyperparameter('wd_TAdam', 1e-5, 1e-1, default_value=1e-5)
        cs.add_hyperparameter(wd_TAdam)

        wd_Adam = UniformFloatHyperparameter('wd_Adam', 1e-5, 1e-1, default_value=1e-5)
        cs.add_hyperparameter(wd_Adam)
        
        wd_SGD = UniformFloatHyperparameter('wd_SGD', 1e-5, 1e-1, default_value=1e-5)
        cs.add_hyperparameter(wd_SGD)

        Cond_lr_scheduler = InCondition(child=lr_scheduler, parent=optim_name, values=['SGD'])
        Cond_momentum = InCondition(child=momentum, parent=optim_name, values=['SGD'])
        Cond_lr_SGD = InCondition(child=lr_SGD, parent=optim_name, values=['SGD'])
        Cond_wd_SGD = InCondition(child=wd_SGD, parent=optim_name, values=['SGD'])
        
        Cond_lr_Adam = InCondition(child=lr_Adam, parent=optim_name, values=['Adam'])
        Cond_wd_Adam = InCondition(child=wd_Adam, parent=optim_name, values=['Adam'])
        
        Cond_lr_Fromage = InCondition(child=lr_Fromage, parent=optim_name, values=['Fromage'])
        Cond_lr_TAdam = InCondition(child=lr_TAdam, parent=optim_name, values=['TAdam'])
        Cond_wd_TAdam = InCondition(child=wd_TAdam, parent=optim_name, values=['TAdam'])
        cs.add_conditions([Cond_lr_scheduler, Cond_momentum, Cond_lr_SGD, Cond_lr_Adam,
                           Cond_lr_Fromage, Cond_lr_TAdam, Cond_wd_SGD, Cond_wd_Adam,Cond_wd_TAdam])

        return cs

            
    def train(self, setting:Configuration):
        print(setting)
        optim_name = setting["optim_name"]
        self.state.optimizer_name = optim_name

        if optim_name == "SGD":
            self.state.Optim_default_params
            self.state.Optim_params['momentum'] = setting["momentum"]
            self.state.Optim_params['learning_rate'] = setting["lr_SGD"]
            self.state.Optim_params['wd'] = setting["wd_SGD"]
            self.state.lr_scheduler_name  = setting["lr_scheduler"]
            self.state.enable_scheduler = True

        elif optim_name == "Adam":
            self.state.Optim_params['learning_rate'] = setting["lr_Adam"]
            self.state.Optim_params['wd'] = setting["wd_Adam"]
            self.state.enable_scheduler = False
        elif optim_name == "TAdam":
            self.state.Optim_params['learning_rate'] = setting["lr_TAdam"]
            self.state.Optim_params['wd'] = setting["wd_TAdam"]
            self.state.enable_scheduler = False
        elif optim_name == "SparseAdam":
            self.state.Optim_params['learning_rate'] = setting["lr_Adam"]
            self.state.enable_scheduler = False
        elif optim_name == "Fromage":
            self.state.Optim_params['learning_rate'] = setting["lr_Fromage"]
            self.state.enable_scheduler = False
            
        ch_name = "prune%s.pth"%(self.state.MODEL_ARCHITECTURES[self.state.args.m])
        self.state.Load_Checkpoint(ch_name)


        self.state.update_optimizer()

        mode = 'No_Save'
        if optim_name == "SGD":
            if self.state.args.training_enabler:
                mode = 'FinalHPO_%s_lr%s_m%s_%s' %(optim_name,setting["lr_SGD"],setting["momentum"],setting["lr_scheduler"])
            if self.state.kfold_enabler:
                test_acc = cross_val.kfold_training(self.state, mode)
            else:
                test_acc = pruning_lib.training(self.state, mode)   
        else:
            if self.state.args.training_enabler:
                mode = 'FinalHPO_%s_lr%s' %(optim_name, self.state.Optim_params['lr'])
            if self.state.kfold_enabler:
                test_acc = cross_val.kfold_training(self.state, mode)
            else:
                test_acc = pruning_lib.training(self.state, mode)  
                logging.info("")


        return 1 - float(test_acc/100)


