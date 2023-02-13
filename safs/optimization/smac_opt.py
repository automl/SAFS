
import logging
import utility
import torch.nn as nn
import settings as settings
import torch.optim as optim
from utility import Set_Lr_Policy
import training.train_lib as train_lib
from tmomentum.optimizers import TAdam
from optimization.fromage import Fromage
from smac.scenario.scenario import Scenario
from ConfigSpace.conditions import InCondition
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.configspace import CategoricalHyperparameter,\
                             UniformFloatHyperparameter,\
                             UniformIntegerHyperparameter




def train_search_fine(setting):
    model = settings.Model
    optim_name = setting.optim_name

    if optim_name == "SGD":

        settings.Optim_default['momentum'] = setting.momentum
        optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                              momentum=settings.Optim_default['momentum'],
                              weight_decay=settings.Optim_default['wd'])

        utility.Set_Lr_Policy(setting.lr_scheduler, optimizer)
        logging.info('SGD : momentum= %s  lr_scheduler=%s' % (setting.momentum,setting.lr_scheduler))

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
        model = train_lib.Training('model_%s_m%s_%s' %(optim_name,settings.Optim_default['momentum'],setting.lr_scheduler), model, criterior, optimizer, settings.DATA_Train)
    else:
        model = train_lib.Training('model_%s' %(optim_name), model, criterior, optimizer, settings.DATA_Train)
    test_loss, test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Test)
    final_test_loss, final_test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Final_Test)
    print('HPO: Test Accuracy, Test Loss =', test_acc, test_loss)
    logging.info('HPO Score: Test Accuracy, Test Loss = %s %s' % (test_acc, test_loss))
    logging.info('Not HPO Score: Final Test Accuracy, Test Loss = %s %s' % (final_test_acc, final_test_loss))
    return 1 - float(test_acc/100)

class run_strategies:
    @staticmethod
    def Run_Smac():
        # build configuration space
        cs = ConfigurationSpace()
        # kernel
        optim_name = CategoricalHyperparameter('optim_name', ['SGD', 'TAdam','Fromage','Adam'], default_value='SGD')#,"SparseAdam"
        cs.add_hyperparameters(optim_name)

        lr_scheduler = CategoricalHyperparameter('lr_scheduler', ['VGG_lr_scheduler', 'Constant', 'Cosine', 'Step', 'Linear', 'ReduceLROnPlateau'], default_value=settings.lr_scheduler_name)
        cs.add_hyperparameter(lr_scheduler)

        momentum = UniformFloatHyperparameter('momentum', 0.1, 0.99, default_value=0.9)
        cs.add_hyperparameters(momentum)

        Cond_lr_scheduler = InCondition(child=lr_scheduler, parent=optim_name, values=['SGD'])
        Cond_momentum = InCondition(child=momentum, parent=optim_name, values=['SGD'])
        cs.add_conditions([Cond_lr_scheduler, Cond_momentum])

        scenario = Scenario(
                {   
                    "run_obj": "quality",
                    "runcount-limit": 20,
                    "cs": cs
                }           )

        #def_value = train_search_fine(cs.get_default_configuration())
        #print("Default Value: %.2f" % def_value)

        smac = SMAC4HPO(scenario=scenario,
                        rng=20,
                        tae_runner=train_search_fine)

        incumbent_configuration = smac.optimize()
        logging.info("best_result: %s" %(incumbent_configuration))











