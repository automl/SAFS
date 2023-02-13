import torch
import lahc
import random
import time
import copy
import logging
import torch.nn as nn
import settings as settings
import utility as utility
import settings as settings
import torch.optim as optim
from simanneal import Annealer
import training.train_lib as train_lib
from pyeasyga.pyeasyga import GeneticAlgorithm


logging = settings.LOGGING 

class Optim_Operation(nn.Module):

    def __init__(self, model=None, criterior=None, optimizer=None):
        super(Optim_Operation, self).__init__()
        self.model = model
        self.criterior = criterior
        self.optimizer = optimizer
        self.genome_size = None
        self.layer_number_optim = None
        self.candidate_activations = ['relu6','hardswish','swish',\
                                      'TanhSoft-1','acon', 'gelu',\
                                      'srs', 'linear', 'relu', 'elu',\
                                      'selu','tanh', 'hardtanh', 'softplus',\
                                      'logsigmod']
#,'hard_elish''elish', 'mish', 'ash''sigmoid''sin', 'cos'

    def search_nodes(self, search_agent=None, input_layer_size=None, layer_number_optim=None):
        self.genome_size = input_layer_size
        activation_temp = [0] * self.genome_size
        self.layer_number_optim = layer_number_optim
        return self.search(search_agent, activation_temp)

    def search_layers(self, search_agent=None, number_of_layers=None):
        self.genome_size = number_of_layers
        activation_temp = [0] * self.genome_size
        return self.search(search_agent, activation_temp)

    def search_networks(self, search_agent=None):
        activation_temp = [0]
        self.genome_size = 1
        return self.search(search_agent, activation_temp)

    def search (self, search_agent, activation_temp):
        if search_agent == 'Random_Assignment':
            for i in range(self.genome_size):
                activation_temp[i] = random.choice(self.candidate_activations)
        elif search_agent == 'GA':
            activation_temp = self.GA()
        elif search_agent == 'LAHC':
            activation_temp = self.LAHC()
        elif search_agent == 'SA':
            activation_temp = self.SA()
        elif search_agent == 'RS':
            activation_temp = self.RS()
        else:
            for i in range(self.genome_size):
                activation_temp[i] = 'relu'
        return activation_temp

    def LAHC(self):
        initial_state = []
        for i in range(self.genome_size):
            initial_state.append(random.choice(self.candidate_activations))

        lahc.LateAcceptanceHillClimber.history_length = settings.LAHC_CONFIG['history_length']
        lahc.LateAcceptanceHillClimber.updates_every = settings.LAHC_CONFIG['updates_every']
        lahc.LateAcceptanceHillClimber.steps_minimum = settings.LAHC_CONFIG['steps_minimum']
        # initializing the problem
        prob = LAHC_Search(initial_state, self.model, self.criterior, self.optimizer,
                           self.candidate_activations, self.layer_number_optim)
        # and run the Late Acceptance Hill Climber for a solution
        prob.run()
        return prob.best_state

    def SA(self):
        initial_state = []
        for i in range(self.genome_size):
            initial_state.append(random.choice(self.candidate_activations))

        tsp = SA_Search(initial_state, self.model, self.criterior, self.optimizer, self.candidate_activations,
                        self.layer_number_optim)
        tsp.Tmax = settings.SA_CONFIG['Tmax']
        tsp.Tmin = settings.SA_CONFIG['Tmin']
        tsp.steps = settings.SA_CONFIG['steps']
        # tsp.set_schedule(tsp.auto(minutes=2))
        # since our state is just a list, slice is the fastest way to copy
        tsp.copy_strategy = "slice"
        state, e = tsp.anneal()
        return state

    def RS(self):
        initial_state = []
        for i in range(self.genome_size):
            initial_state.append(random.choice(self.candidate_activations))

        rs = RS_Search(initial_state, self.model, self.criterior, self.optimizer, self.candidate_activations,
                       self.layer_number_optim)
        return rs.run()

    def GA(self):
        data = [None] * self.genome_size
        _ga = GA_Search(data, self.model, self.criterior, self.optimizer, self.candidate_activations,
                        self.layer_number_optim)
        return _ga._run()


class GA_Search():

    def __init__(self, data, model, criterior, optimizer, candidate_activations, layer_number_optim):
        self.criterior = criterior
        self.model = model
        self.optimizer = optimizer
        self.layer_number_optim = layer_number_optim
        self.data = data
        self.candidate_activations = candidate_activations

    def _run(self):
        ga = GeneticAlgorithm(self.data, settings.GA_CONFIG['population_size'], settings.GA_CONFIG['generations'],
                              settings.GA_CONFIG['crossover_probability'], settings.GA_CONFIG['mutation_probability'],
                              settings.GA_CONFIG['elitism'], settings.GA_CONFIG['maximise_fitness'])
        ga.create_individual = self.create_individual
        ga.fitness_function = self.fitness

        start = time.time()
        ga.run()
        end = time.time()
        print("Elapsed GA Time is Sec:", (end - start))
        print(ga.best_individual())
        return ga.best_individual()[1]

    def create_individual(self, data):
        temp = []
        for i in range(len(data)):
            temp.append(random.choice(self.candidate_activations))
        return temp

    def fitness(self, individual, data):
        counter = 0
        settings.TRAINING_ENABLER = False
        if settings.OPTIM_MODE == 0:
            self.model.optim_dictionary[self.layer_number_optim][2] = individual
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training(self.model, self.criterior, self.optimizer),
                                                       self.criterior)
            self.fitness_value = test_acc
        elif settings.OPTIM_MODE == 1:
            for i in self.model.optim_dictionary:
                self.model.optim_dictionary[i][2] = individual[counter]
                counter += 1
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training(self.model, self.criterior, self.optimizer),
                                                       self.criterior)
            self.fitness_value = test_acc
        else:
            for i in self.model.optim_dictionary:
                self.model.optim_dictionary[i][2] = individual
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training(self.model, self.criterior, self.optimizer),
                                                       self.criterior)
            self.fitness_value = test_acc
        return self.fitness_value


class LAHC_Search(lahc.LateAcceptanceHillClimber):

    def __init__(self, initial_state, model, criterior, optimizer, candidate_activations, layer_number_optim):
        super(LAHC_Search, self).__init__(initial_state=initial_state)
        self.fitness_value = 0
        self.train_loader = settings.DATA_Train
        self.test_loader = settings.DATA_Test
        self.criterior = criterior
        self.optimizer = optimizer
        self.model = model
        self.layer_number_optim = layer_number_optim
        self.candidate_activations = candidate_activations

    def move(self):
        c = random.randint(0, len(self.state) - 1)
        self.state[c] = random.choice(self.candidate_activations)
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        logging = settings.LOGGING 

        counter = 0
        self.train_loader = settings.DATA_Train
        test_loader = settings.DATA_Test
        settings.TRAINING_ENABLER = False
        # load prune model weights
        self.model, self.optimizer, _, _ = utility.Load_Checkpoint(self.model, self.optimizer,
                                                                   'Model_prune' +
                                                                   settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] + '.pth')

        self.optimizer = optim.SGD(self.model.parameters(), lr=settings.Optim_default['learning_rate'],
                              momentum=settings.Optim_default['momentum'],
                              weight_decay=settings.Optim_default['wd'])
        utility.Set_Lr_Policy(settings.lr_scheduler_name, self.optimizer)
       #TODO: define optimizer


        if settings.OPTIM_MODE == 0:
            self.model.optim_dictionary[self.layer_number_optim][2] = self.state
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training('No_Save', self.model, self.criterior, self.optimizer, self.train_loader),
                                                       self.criterior, settings.DATA_Test)
            logging.info(str(self.model.optim_dictionary))
            logging.info("test_loss: %s ,  test_acc:%s " %(test_loss, test_acc))
            self.fitness_value = test_acc
        elif settings.OPTIM_MODE == 1:
            for i in self.model.optim_dictionary:
                self.model.optim_dictionary[i][2][0] = self.state[counter]
                counter += 1
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training('No_Save', self.model, self.criterior, self.optimizer, self.train_loader),
                                                       self.criterior, settings.DATA_Test)
            logging.info(str(self.model.optim_dictionary)) #TODO:
            logging.info("test_loss: %s ,  test_acc:%s " %(test_loss, test_acc))
            self.fitness_value = test_acc
        else:
            for i in self.model.optim_dictionary:
                self.model.optim_dictionary[i][2] = self.state
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training('No_Save', self.model, self.criterior, self.optimizer, self.train_loader),
                                                       self.criterior, settings.DATA_Test)
            logging.info(str(self.model.optim_dictionary)) 
            logging.info("test_loss: %s ,  test_acc:%s " %(test_loss, test_acc))
            self.fitness_value = test_acc

        return 1/self.fitness_value

    @property
    def exact_solution(self):
        return [self.fitness_value]


class SA_Search(Annealer):

    def __init__(self, state, model, criterior, optimizer, candidate_activations, layer_number_optim):
        self.criterior = criterior
        self.model = model
        self.optimizer = optimizer
        self.layer_number_optim = layer_number_optim
        self.candidate_activations = candidate_activations
        super(SA_Search, self).__init__(state)

    def move(self):
        c = random.randint(0, len(self.state) - 1)
        self.state[c] = random.choice(self.candidate_activations)
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        counter = 0
        settings.TRAINING_ENABLER = False
        if settings.OPTIM_MODE == 0:
            self.model.optim_dictionary[self.layer_number_optim][2] = self.state
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training(self.model, self.criterior, self.optimizer),
                                                       self.criterior)
            self.fitness_value = test_acc
        elif settings.OPTIM_MODE == 1:
            for i in self.model.optim_dictionary:
                self.model.optim_dictionary[i][2] = self.state[counter]
                counter += 1
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training(self.model, self.criterior, self.optimizer),
                                                       self.criterior)
            self.fitness_value = test_acc
        else:
            for i in self.model.optim_dictionary:
                self.model.optim_dictionary[i][2] = self.state
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training(self.model, self.criterior, self.optimizer),
                                                       self.criterior)
            self.fitness_value = test_acc
        return 1/self.fitness_value


class RS_Search():

    def __init__(self, initial_state, model, criterior, optimizer, candidate_activations, layer_number_optim):
        self.criterior = criterior
        self.model = model
        self.candidate_activations = candidate_activations
        self.optimizer = optimizer
        self.layer_number_optim = layer_number_optim
        self.iteration_search = settings.RS_CONFIG['iteration_search']
        self.prev_state = initial_state
        self.best_state = []

    def run(self):
        i = 0
        energy_prev_state = self.fitness(self.prev_state)
        while i < self.iteration_search:
            new_state = self.move(self.prev_state)
            energy_new_state = self.fitness(new_state)
            if energy_prev_state < energy_new_state:
                self.prev_state = new_state
                energy_prev_state = energy_new_state
                self.best_state = copy.deepcopy(new_state)
            i += 1
        return self.best_state

    def move(self, state):
        a = random.randint(0, len(state) - 1)
        state[a] = random.choice(self.candidate_activations)
        b = random.randint(0, len(state) - 1)
        c = random.randint(0, len(state) - 1)
        state[b], state[c] = state[c], state[b]
        return state

    def fitness(self, state):
        counter = 0
        settings.TRAINING_ENABLER = False
        if settings.OPTIM_MODE == 0:
            self.model.optim_dictionary[self.layer_number_optim][2] = state
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training(self.model, self.criterior, self.optimizer),
                                                       self.criterior)
        elif settings.OPTIM_MODE == 1:
            for i in self.model.optim_dictionary:
                self.model.optim_dictionary[i][2] = state[counter]
                counter += 1
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training(self.model, self.criterior, self.optimizer),
                                                       self.criterior)
        else:
            for i in self.model.optim_dictionary:
                self.model.optim_dictionary[i][2] = state
            test_loss, test_acc = train_lib.Test_Model(train_lib.Training(self.model, self.criterior, self.optimizer),
                                                       self.criterior)
        return test_acc


def AF_Operation_Optimization(model, criterior, optimizer, train_loader):
    start = time.time()
    activation_temp = None
    optim_method = settings.OPTIM_METHODS[settings.OPTIM_METHOD]
    optim_agent = Optim_Operation(model, criterior, optimizer)

    if settings.OPTIM_MODE == 0:
        if optim_method == 'LAHC':
            for i in model.optim_dictionary:
                activation_temp = optim_agent.search_nodes(optim_method,
                                                             input_layer_size=model.optim_dictionary[i][1],
                                                             layer_number_optim=i)
                model.optim_dictionary[i][2] = activation_temp
        elif optim_method == 'RS':
            for i in model.optim_dictionary:
                activation_temp = optim_agent.search_nodes(optim_method,
                                                             input_layer_size=model.optim_dictionary[i][1],
                                                             layer_number_optim=i)
                model.optim_dictionary[i][2] = activation_temp
        elif optim_method == 'GA':
            for i in model.optim_dictionary:
                activation_temp = optim_agent.search_nodes(optim_method,
                                                             input_layer_size=model.optim_dictionary[i][1],
                                                             layer_number_optim=i)[1]
                model.optim_dictionary[i][2] = activation_temp
        elif optim_method == 'SA':
            for i in model.optim_dictionary:
                activation_temp = optim_agent.search_nodes(optim_method,
                                                             input_layer_size=model.optim_dictionary[i][1],
                                                             layer_number_optim=i)
                model.optim_dictionary[i][2] = activation_temp

    elif (settings.OPTIM_MODE == 1):
        counter = 0
        if optim_method == 'LAHC':
            activation_temp = optim_agent.search_layers(optim_method,
                                                        number_of_layers=len(model.optim_dictionary))
            for i in model.optim_dictionary:
                model.optim_dictionary[i][2][0] = activation_temp[counter]
                counter += 1
        elif optim_method == 'RS':
            activation_temp = optim_agent.search_layers(optim_method,
                                                        number_of_layers=len(model.optim_dictionary))
            for i in model.optim_dictionary:
                model.optim_dictionary[i][2] = activation_temp[counter]
                counter += 1
        elif optim_method == 'GA':
            activation_temp = optim_agent.search_layers(optim_method,
                                                        number_of_layers=len(model.optim_dictionary))
            for i in model.optim_dictionary:
                model.optim_dictionary[i][2] = activation_temp[counter]
                counter += 1
        elif optim_method == 'SA':
            activation_temp = optim_agent.search_layers(optim_method,
                                                        number_of_layers=len(model.optim_dictionary))
            for i in model.optim_dictionary:
                model.optim_dictionary[i][2] = activation_temp[counter]
                counter += 1

    else:
        activation_temp = optim_agent.search_networks(optim_method)
        for i in model.optim_dictionary:
            model.optim_dictionary[i][2] = activation_temp

    ''' 
    end = time.time()
    print (activation_temp)
    model_temp.layers[1].activations = activation_temp

    test_loss, test_acc = train_lib.Test_Model(model_temp, criterior)
    print('Test Loss / Accuracy with Optimization', optim_method, test_loss, test_acc)
    print("Elapsed  Time is Sec:", (end - start))
    print('-' * 20)
    save_file = 'saved_models/'+'Model_'+optim_method+'.pth'
    torch.save(model_temp, save_file)
    '''

    return model
