import utility
import logging
import settings as settings
import torch.optim as optim
import training.train_lib as train_lib
from skopt.utils import use_named_args
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer

# config variables for HPO problems
learning_rate = Real(low=1e-3, high=2, prior='log-uniform', name='learning_rate')
momentum = Real(low=0.1, high=1.5, name='momentum')
wd = Real(low=5e-5, high=5e-3, prior='log-uniform', name='wd')
lr_scheduler = Categorical(categories=["Constant", "Cosine", "Step", "Linear", "ReduceLROnPlateau"], name='lr_scheduler')
dimensions = [learning_rate,
              momentum,
              wd,
              lr_scheduler]

factor_R = Real(low=0.001, high=0.2, prior='log-uniform', name='factor_R')
patience = Integer(1, 10, name='patience')
min_lr = Real(low=1e-6, high=0.002, prior='log-uniform', name='min_lr')
dimensions_RLP = [factor_R,
                    patience,
                    min_lr]

step_size = Integer(10, 140, name='step_size')
gamma = Real(low=0.01, high=0.3, prior='log-uniform', name='gamma')
dimensions_Step = [step_size,
                    gamma]

factor_C = Real(low=1e-4, high=2, prior='log-uniform', name='factor_C')
total_iters = Integer(1, 30, name='total_iters')
dimensions_Const = [factor_C,
                    total_iters]

start_factor = Real(low=0.001, high=1, prior='log-uniform', name='start_factor')
total_iters = Integer(1, 10, name='total_iters')
dimensions_Linear = [start_factor,
                    total_iters]

T_max = Integer(10, 2000, name='T_max')
eta_min = Integer(0, 30, name='eta_min')
dimensions_Cos = [T_max,
                  eta_min]

# Objective functions
@use_named_args(dimensions=dimensions_Linear)
def HPO_search_Linear(start_factor, total_iters):
    model = settings.Model
    criterior = settings.Criterior
    optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                                            momentum=settings.Optim_default['momentum'],
                                            weight_decay=settings.Optim_default['wd'])
    settings.LR_SCHEDULER_CONFIG['Linear'] =\
        {'start_factor': start_factor, 'total_iters': total_iters}
    utility.Set_Lr_Policy('Linear', optimizer)
    # logging.info('lr scheduler policy = %s' % lr_scheduler)
    # settings.TRAIN_EPOCH = 1  # TODO:
    settings.TRAINING_ENABLER = 1
    model = train_lib.Training('x', model, criterior, optimizer, settings.DATA_Train)
    test_loss, test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Test)
    print('HPO: Linear, Test Loss =', test_acc, test_loss)
    return -float(test_acc / 100)

@use_named_args(dimensions=dimensions_RLP)
def HPO_search_RLP(factor_R, patience, min_lr):
    model = settings.Model
    criterior = settings.Criterior
    optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                                            momentum=settings.Optim_default['momentum'],
                                            weight_decay=settings.Optim_default['wd'])
    settings.LR_SCHEDULER_CONFIG['ReduceLROnPlateau'] =\
        {'factor': factor_R, 'patience': patience, 'min_lr': min_lr}
    utility.Set_Lr_Policy('ReduceLROnPlateau', optimizer)
    # logging.info('lr scheduler policy = %s' % lr_scheduler)
    # settings.TRAIN_EPOCH = 1  # TODO:
    settings.TRAINING_ENABLER = 1
    model = train_lib.Training('x', model, criterior, optimizer, settings.DATA_Train)
    test_loss, test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Test)
    print('HPO: ReduceLROnPlateau, Test Loss =', test_acc, test_loss)
    return -float(test_acc / 100)

@use_named_args(dimensions=dimensions)
def HPO_search_Const(factor_C, total_iters):
    model = settings.Model
    criterior = settings.Criterior
    optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                                            momentum=settings.Optim_default['momentum'],
                                            weight_decay=settings.Optim_default['wd'])
    settings.LR_SCHEDULER_CONFIG['Constant'] =\
        {'factor': factor_C, 'total_iters': total_iters}
    utility.Set_Lr_Policy('Constant', optimizer)
    # logging.info('lr scheduler policy = %s' % lr_scheduler)
    # settings.TRAIN_EPOCH = 1  # TODO:
    settings.TRAINING_ENABLER = 1
    model = train_lib.Training('x', model, criterior, optimizer, settings.DATA_Train)
    test_loss, test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Test)
    print('HPO: Constant, Test Loss =', test_acc, test_loss)
    return -float(test_acc / 100)

@use_named_args(dimensions=dimensions)
def HPO_search_Cos(T_max, eta_min):
    model = settings.Model
    criterior = settings.Criterior
    optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                                            momentum=settings.Optim_default['momentum'],
                                            weight_decay=settings.Optim_default['wd'])
    settings.LR_SCHEDULER_CONFIG['Cosine'] =\
        {'T_max': T_max, 'eta_min': eta_min}
    utility.Set_Lr_Policy('Cosine', optimizer)
    # logging.info('lr scheduler policy = %s' % lr_scheduler)
    # settings.TRAIN_EPOCH = 1  # TODO:
    settings.TRAINING_ENABLER = 1
    model = train_lib.Training('x', model, criterior, optimizer, settings.DATA_Train)
    test_loss, test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Test)
    print('HPO: Cosine, Test Loss =', test_acc, test_loss)
    return -float(test_acc / 100)

@use_named_args(dimensions=dimensions_Step)
def HPO_search_Step(step_size, gamma):
    model = settings.Model
    criterior = settings.Criterior
    optimizer = optim.SGD(model.parameters(), lr=settings.Optim_default['learning_rate'],
                                            momentum=settings.Optim_default['momentum'],
                                            weight_decay=settings.Optim_default['wd'])
    settings.LR_SCHEDULER_CONFIG['Step'] =\
        {'step_size': step_size, 'gamma': gamma}
    utility.Set_Lr_Policy('Step', optimizer)
    model = train_lib.Training('x', model, criterior, optimizer, settings.DATA_Train)
    test_loss, test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Test)
    print('HPO: Step, Test Loss =', test_acc, test_loss)
    return -float(test_acc / 100)

@use_named_args(dimensions=dimensions)
def train_search(learning_rate, momentum, wd, lr_scheduler):
    model = settings.Model
    criterior = settings.Criterior
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=wd)
    logging.info('lr scheduler policy = %s' % lr_scheduler)
    utility.Set_Lr_Policy(lr_scheduler, optimizer)
    model = train_lib.Training('x', model, criterior, optimizer, settings.DATA_Train)
    test_loss, test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Test)
    print('HPO: Test Accuracy, Test Loss =', test_acc, test_loss)
    return -float(test_acc / 100)

@use_named_args(dimensions=dimensions)
def train_search_fine(learning_rate, momentum, wd, lr_scheduler):
    model = settings.Model
    criterior = settings.Criterior
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=wd)
    logging.info('lr scheduler policy = %s' % lr_scheduler)
    utility.Set_Lr_Policy(lr_scheduler, optimizer)
    settings.TRAINING_ENABLER = 1
    model = train_lib.Training('x', model, criterior, optimizer, settings.DATA_Train)
    test_loss, test_acc = train_lib.Test_Model(model, criterior, settings.DATA_Test)
    print('Full train after HPO: Test Accuracy, Test Loss =', test_acc, test_loss)
    return -float(test_acc / 100)

# Run functions used as main code for search HPO for specific task
def Run_RLP():
    default_parameters_RLP = [0.1, 5, 0.001]
    # %%time
    search_result_RLP = gp_minimize(func=HPO_search_RLP,
                                    dimensions=dimensions_RLP,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=40,
                                    x0=default_parameters_RLP)
    RLP_result = search_result_RLP.x
    print('search_result_RLP.x:', search_result_RLP.x)
    print('search_result_RLP.fun:', search_result_RLP.fun)
    print(sorted(zip(search_result_RLP.func_vals, search_result_RLP.x_iters)))
    settings.LR_SCHEDULER_CONFIG['ReduceLROnPlateau'] = \
        {'factor': RLP_result[0], 'patience': RLP_result[1], 'min_lr': RLP_result[2]}

def Run_Const():
    default_parameters_Const = [0.05, 0.9]
    # %%time
    search_result_Const = gp_minimize(func=HPO_search_Const,
                                      dimensions=dimensions_Const,
                                      acq_func='EI',  # Expected Improvement.
                                      n_calls=40,
                                      x0=default_parameters_Const)
    Const_result = search_result_Const.x
    print('search_result_Const.x:', search_result_Const.x)
    print('search_result_Const.fun:', search_result_Const.fun)
    print(sorted(zip(search_result_Const.func_vals, search_result_Const.x_iters)))
    settings.LR_SCHEDULER_CONFIG['Constant'] = \
        {'factor': Const_result[0], 'total_iters': Const_result[1]}

def Run_Cosine():
    default_parameters_Cos = [1000, 0]
    # %%time
    search_result_Cos = gp_minimize(func=HPO_search_Cos,
                                    dimensions=dimensions_Cos,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=40,
                                    x0=default_parameters_Cos)
    Cos_result = search_result_Cos.x
    print('search_result_RLP.x:', search_result_Cos.x)
    print('search_result_RLP.fun:', search_result_Cos.fun)
    print(sorted(zip(search_result_Cos.func_vals, search_result_Cos.x_iters)))
    settings.LR_SCHEDULER_CONFIG['Cosine'] = \
        {'T_max': Cos_result[0], 'eta_min': Cos_result[1]}

def Run_Step():
    default_parameters_Step = [70, 0.1]
    # %%time
    search_result_Step = gp_minimize(func=HPO_search_Step,
                                     dimensions=dimensions_Step,
                                     acq_func='EI',  # Expected Improvement.
                                     n_calls=40,
                                     x0=default_parameters_Step)
    Step_result = search_result_Step.x
    print('search_result_Step.x:', search_result_Step.x)
    print('search_result_Step.fun:', search_result_Step.fun)
    print(sorted(zip(search_result_Step.func_vals, search_result_Step.x_iters)))
    settings.LR_SCHEDULER_CONFIG['Step'] = \
        {'step_size': Step_result[0], 'gamma': Step_result[1]}

def Run_Linear():
    default_parameters_Linear = [0.333, 5]
    # %%time
    search_result_Linear = gp_minimize(func=HPO_search_Linear,
                                       dimensions=dimensions_Linear,
                                       acq_func='EI',  # Expected Improvement.
                                       n_calls=40,
                                       x0=default_parameters_Linear)
    Linear_result = search_result_Linear.x
    print('search_result_Linear.x:', search_result_Linear.x)
    print('search_result_Linear.fun:', search_result_Linear.fun)
    print(sorted(zip(search_result_Linear.func_vals, search_result_Linear.x_iters)))
    settings.LR_SCHEDULER_CONFIG['Linear'] = \
        {'start_factor': Linear_result[0], 'total_iters': Linear_result[1]}

class run_strategies:
    @staticmethod
    def run_search_params(model):

        settings.TRAINING_ENABLER = False
#################################   RLP     ########################################
        # Run_RLP()
#################################   Const   ########################################
        # Run_Const()
#################################   Cosine  ########################################
        #Run_Cosine()
#################################   step    ########################################
        # Run_Step()
#################################   Linear  ########################################
        # Run_Linear()
#################################   Final   ########################################
        default_parameters = [0.1, 0.9, 5e-4, 'Step']
        # %%time
        search_result = gp_minimize(func=train_search,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=40,
                                    x0=default_parameters)
        print('search_result.x:', search_result.x)
        print('search_result.fun:', search_result.fun)
        print(sorted(zip(search_result.func_vals, search_result.x_iters)))
        print('Final_search_result.x', search_result.x)
        settings.TRAINING_ENABLER = True
        acc = train_search_fine(x=search_result.x)
        print('final_acc:', acc)
        return model


