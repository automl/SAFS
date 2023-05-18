import torch
import utility
import time
import torch.optim as optim
from tmomentum.optimizers import TAdam
from optimization.fromage import Fromage
from lion_pytorch import Lion
from models.localvit import create_args_localvit
from timm.optim import create_optimizer

LOG_DIR = 'saved_logs/logs/'
LOG_Text = LOG_DIR+'Output_Log_Text.txt'

SEED = 42
seed_enabler = None

OPTIM_MODE = 1




GA_CONFIG = {
    "population_size": 5,
    "generations": 2,
    "crossover_probability": 0.8,
    "mutation_probability": 0.3,
    "elitism": True,
    "maximise_fitness": True
}

class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """
    def __init__(self, args):
        self.epoch = -1
        self.grad_flow_enabler = False
        self.model = None
        self.DDP_model = None
        self.optimizer = None
        self.optimizer_name = "SGD"
        self.data_pack = None
        self.criterior = None
        self.test_final = False
        self.args = args
        self.kfold_enabler = 1
        self.kfold_indexes = None
        self.kfold_indexes_dense = None
        self.rank_id = None
        #self.logging = logging
        self.test_acc_best = []
        self.enable_scheduler = 1
        self.device_id = None
        self.lr_scheduler_name = None
        #self.lr_scheduler = None
        self.CHECKPOINT_PATH = None
        self.CHECKPOINT_dense_PATH = None
        self.MODEL_ARCHITECTURES = {
                                        0: '1-Layer MLP',
                                        1: 'Lenet5',
                                        2: 'VGG-16',
                                        3: 'ResNet-18',
                                        4: 'EfficientNet_B0',
                                        5: 'LocalVit'
                                    }
        self.LR_SCHEDULER_CONFIG = {
                                    'ReduceLROnPlateau': {'factor': 0.05, 'patience': 2, 'min_lr': 0,
                                    'threshold':0.0001, 'cooldown':0,'eps':1e-08,'threshold_mode':'rel'},
                                    'Constant': {'factor': 0.1, 'total_iters': 5},
                                    'Cosine': {'T_max': 1000, 'eta_min': 0},
                                    'Step': {'step_size': 70, 'gamma': 0.1},
                                    'Linear': {'start_factor': 0.1, 'total_iters': 5}
                                }
        self.Optim_default_params = {'learning_rate': 0.1, 'momentum': 0.9, 'wd': 5e-4}
        self.Optim_params = {'learning_rate': 0.1, 'momentum': 0.9, 'wd': 5e-4}
        self.requires_grad_enabler = False
        self.requires_grad_data = {"disable":  {"relu":0,'relu6':0,'hardswish':0,'swish':0,\
                                                  'TanhSoft-1':2,'acon':3, 'gelu':0,\
                                                  'srs':2, 'linear':0, 'elu':0,\
                                                  'selu':0,'tanh':0, 'hardtanh':0, 'softplus':0,\
                                                  'logsigmiod':0, 'symlog':0,'symexp':0, 'SRelu':4,\
                                                  'FALU':2},
                                    
                                      "enable":    {"relu":2,'relu6':2,'hardswish':2,'swish':2,\
                                                   'TanhSoft-1':2,'acon':3, 'gelu':2,\
                                                   'srs':2, 'linear':1, 'elu':2,\
                                                   'selu':2,'tanh':2, 'hardtanh':2, 'softplus':2,\
                                                   'logsigmiod':2,'symlog':2,'symexp':2, 'SRelu':4,\
                                                    'FALU':2 }
                                    }  
        self.OPTIM_MODES = {
                                0: 'Node',
                                1: 'Layer',
                                2: 'Network'
                            }    
        self.OPTIM_METHODS = {
                                0: 'LAHC',
                                1: 'RS',
                                2: 'GA',
                                3: 'SA'
                            }
        self.LAHC_CONFIG = {
                                "history_length": 3,
                                "updates_every": 100000,
                                "steps_minimum": args.activ_search_steps
                            }
        self.DATASETS = {
                        0:'MNIST',
                        1:'CIFAR10',
                        2:'Imagenet16'
                        }
        self.RS_CONFIG = {
                           "iteration_search": args.activ_search_steps
                         }
        self.SA_CONFIG = {
                            "Tmax": 10000,
                            "Tmin": 10,
                            "steps": args.activ_search_steps
                         }

             
    def prepare_dataset(self,state_, rankid):
            args = self.args
            if args.d == 0:
                kfold_data, train_loader, test_loader, final_test_loader = utility.MNIST_Loaders(state_, rankid)
            elif args.d == 1:
                kfold_data, train_loader, test_loader, final_test_loader = \
                    utility.CIFAR10_Loaders(state_, rankid)
            elif args.d == 2:
                kfold_data, train_loader, test_loader, final_test_loader = \
                utility.IMGN_loader(state_, rankid)
            self.data_pack = {"kfold_dataset":kfold_data, "train_loader":train_loader,\
                "test_loader":test_loader, "final_test_loader":final_test_loader}

    def update_optimizer(self):
        name = self.optimizer_name
        if name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.Optim_params['learning_rate'],
                            momentum=self.Optim_params['momentum'],
                            weight_decay=self.Optim_params['wd'])
        elif name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.Optim_params['learning_rate'],\
                weight_decay=self.Optim_params['wd'], betas=(0.9, 0.99))
        elif name == "TAdam":
            self.optimizer = TAdam(self.model.parameters(), lr=self.Optim_params['learning_rate'],\
                weight_decay=self.Optim_params['wd'], betas=(0.9, 0.99))
        elif name == "Fromage":
            self.optimizer = Fromage(self.model.parameters(), lr=self.Optim_params['learning_rate'])
        elif name == "localvit":
            localvit_args = create_args_localvit(self)
            self.optimizer = create_optimizer(localvit_args, self.model)
        elif name == "Lion":
            self.optimizer = Lion(self.model.parameters(), \
                                  lr = self.Optim_params['learning_rate'], \
                                    weight_decay = self.Optim_params['wd'])
    
    
    def show_optim_dictionary(self):
        keys = list(self.model.optim_dictionary.keys())
        return [self.model.optim_dictionary[item][2][0] for i, item in enumerate(keys)]



    def save(self, f):
        torch.save(self.capture_snapshot(), f, _use_new_zipfile_serialization=False)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"{device_id}")
        self.apply_snapshot(snapshot, device_id)
        
    def Load_Checkpoint(self, checkpoint_name):
        # Load the training model

        if 'prune' in checkpoint_name: #TODO:
            checkpoint = torch.load(self.CHECKPOINT_PATH + checkpoint_name)
        elif 'first_train' in checkpoint_name:
            checkpoint = torch.load(self.CHECKPOINT_dense_PATH + checkpoint_name)
            for item in self.model.optim_dictionary.keys():
                self.model.optim_dictionary[item][2][0] = checkpoint['optim_dictionary'][item][2][0]
            self.kfold_indexes_dense =  checkpoint['kfold_indexes']
        else :
            checkpoint = torch.load(self.CHECKPOINT_PATH + checkpoint_name)
            for item in self.model.optim_dictionary.keys():
                self.model.optim_dictionary[item][2][0] = checkpoint['optim_dictionary'][item][2][0]
           
        self.model.load_state_dict(checkpoint['model_state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.kfold_indexes =  checkpoint['kfold_indexes']
    
    def Create_Checkpoint(self, results):
        # Create the training checkpoint
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_dictionary': self.model.optim_dictionary,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'results': results,
            'kfold_indexes' : self.kfold_indexes
        }
        return checkpoint

    def Save_Checkpoint(self, checkpoint, checkpoint_name):
        # Save the training model
        if 'first_train' in checkpoint_name:
            torch.save(checkpoint, self.CHECKPOINT_dense_PATH + checkpoint_name)
        else:
            torch.save(checkpoint, self.CHECKPOINT_PATH + checkpoint_name)
    def save_dataset(self):
        torch.save(self.data_pack["kfold_dataset"], self.CHECKPOINT_dense_PATH+str(self.args.m)+str(self.args.d)+"saved_dataset.pth")
    def load_dataset(self):
        self.data_pack["kfold_dataset"] = torch.load(self.CHECKPOINT_dense_PATH+str(self.args.m)+str(self.args.d)+"saved_dataset.pth")        
