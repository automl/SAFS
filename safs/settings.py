######################################              Flages               #############################################
# True: enable train  False: use saved checkpoints
Finetune_a_b = True
First_train = True
First_stage = False
Second_stage = False
Train_after_prune = False
Enable_scheduler = True

run_activation_ab_flag = None
run_activation = None
run_mode = None
google_account = None
set_device = None
######################################  Log/Checkpoint Path Configuration #############################################

#directory to save weights file (this should be a fucntion)
CHECKPOINT_PATH = 'saved_checkpoints/'
CHECKPOINT_Load_PATH = 'saved_checkpoints/'

save_id = None
CHECKPOINT = False
LOG_DIR = 'saved_logs/logs/'
LOG_Text = LOG_DIR+'Output_Log_Text.txt'
LOGGING = None
#######################################  Pruning Configuration ########################################################
Pruning_Type = 'l1_unstructured'
# ln_structured or l1_unstructured
#######################################  Device Configuration  ########################################################
GPU_ENABLED = True
#######################################  Optimization Configuration  ##################################################
OPTIM_EPOCH = 80
SEED = 42
OPTIM_MODE = 0
OPTIM_METHODS = {
    0: 'LAHC',
    1: 'RS',
    2: 'GA',
    3: 'SA',
    4: 'Random_Assignment'
}
OPTIM_METHOD = 0 # 0:LAHC is the default search method

OPTIM_MODES = {
    0: 'Node',
    1: 'Layer',
    2: 'Network'
}

LAHC_CONFIG = {
    "history_length": 3,
    "updates_every": 100000,
    "steps_minimum": 20
}

RS_CONFIG = {
"iteration_search": 10
}

SA_CONFIG = {
    "Tmax": 10000,
    "Tmin": 10,
    "steps": 10
}

GA_CONFIG = {
    "population_size": 5,
    "generations": 2,
    "crossover_probability": 0.8,
    "mutation_probability": 0.3,
    "elitism": True,
    "maximise_fitness": True
}

OPTIM_STAGE = 'Stage_1'

#######################################  Pruning Configuration  ##################################################
PRUNED_DENSE = 'Dense'
Pruning_Rate = None
#######################################  Training Configuration  ######################################################
TRAINING_ENABLER = True
BATCH_SIZE = 2048
WORKERS = 4
TRAIN_EPOCH = 200
MODEL_ARCHITECTURES = {
    0: '1-Layer MLP',
    1: 'Lenet5',
    2: 'VGG-16',
    3: 'ResNet-18',
    4:"EfficientNet_B0"
}

MODEL_ARCHITECTURE = 2
lr_scheduler_name = None
LR_SCHEDULER = None
Optim_default = {'learning_rate': 0.1, 'momentum': 0.9, 'wd': 5e-4}

LR_SCHEDULER_CONFIG = {
    'ReduceLROnPlateau': {'factor': 0.05, 'patience': 2, 'min_lr': 0,
     'threshold':0.0001, 'cooldown':0,'eps':1e-08,'threshold_mode':'rel'},
    'Constant': {'factor': 0.333, 'total_iters': 5},
    'Cosine': {'T_max': 1000, 'eta_min': 0},
    'Step': {'step_size': 70, 'gamma': 0.1},
    'Linear': {'start_factor': 0.333, 'total_iters': 5}
}

DATASETS = {
    0: 'MNIST',
    1: 'CIFAR10'
}
DATASET = 1

#######################################  Global save Config and Data ##################################################

DATA_Train = None
DATA_Final_Test = None
All_Dataset = None
DATA_Test = None
Criterior = None
Optimizer = None
# only use for HPO #TODO:
Model = None



