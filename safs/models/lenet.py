'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn import Module
import utility
import settings
from models.activation import Activation_Function

class Conv_Layer_Activation (nn.Module):

    def __init__(self, activations=None, name=None, size=None):
        super(Conv_Layer_Activation, self).__init__()
        self.layer_type = 'Conv_Layer_Activation'
        self.activations = activations
        self.acon_size = size
        self.name = name
        self.activation = Activation_Function(self.name ,self.acon_size)
        self.activation_size = None

    def forward(self, x):
        result = torch.zeros(x.size())
        if torch.cuda.is_available():
            result = result.cuda()

        reshape_input = x.view(x.size()[0], -1)
        self.activation_size = reshape_input.size()[1]
        result_temp = torch.zeros(reshape_input.size())
        if torch.cuda.is_available():
            result_temp = result_temp.cuda()

        if settings.OPTIM_MODE == 0: #TODO:
            for j in range(self.activation_size):
                result_temp[:, j] = self.activation(reshape_input[:, j], self.activations[j])
        else:
            result_temp = self.activation(reshape_input, self.activations[0])

        result = result_temp.view(x.size())
        return result

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.optim_dictionary = {
            1: ['1', 65536, []],
            2: ['2', 65536, []],
            3: ['3', 32768, []],
            4: ['4', 32768, []],

        }
        self.optim_dictionary[1][2].append('relu')
        self.optim_dictionary[2][2].append('relu')
        self.optim_dictionary[3][2].append('relu')
        self.optim_dictionary[4][2].append('relu')


        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = Conv_Layer_Activation(activations=self.optim_dictionary[1][2])#nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = Conv_Layer_Activation(activations=self.optim_dictionary[2][2])#nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = Conv_Layer_Activation(activations=self.optim_dictionary[3][2])#nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = Conv_Layer_Activation(activations=self.optim_dictionary[4][2])#nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return y

if __name__ == "__main__":

    # net.named_parameters() 也是可迭代对象，既能调出网络的具体参数，也有名字信息
    # for name, parameters in lenet.named_parameters():
    #     print(name, ';', parameters.size())

    net = LeNet()
    print(net.name)
    summary(net, input_size=(3, 32, 32), device='cpu')