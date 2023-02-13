import torch
import torch.nn as nn
import math
import numpy
import torch.nn.functional as F
import utility 
import settings as settings
from models.activation import Activation_Function


class MLP(nn.Module):
    def __init__(self, size_list):
        super(MLP, self).__init__()
        self.layers = []
        self.activations = []
        self.model_type = 'MLP'
        self.optim_dictionary = {1: ['Classification_Layer_Activation', 120, []]}

        if settings.OPTIM_MODE == 0:
            for node in range(size_list[1]):
                self.optim_dictionary[1][2].append('relu')
        else:# layer wise AF optimization
            self.optim_dictionary[1][2].append('relu')

        self.size_list = size_list
        for i in range(len(size_list) - 2):
            self.layers.append(nn.Linear(size_list[i], size_list[i + 1]))
            self.layers.append(Classification_Layer_Activation(activations=self.optim_dictionary[1][2]))
        self.layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*self.layers)

    def forwards(self, x):
        x = x.view(-1, self.size_list[0])
        return self.net(x)


class LeNet5(nn.Module):

    #LeNet5 model

    def __init__(self, num_classes = 10):
        super(LeNet5, self).__init__()
        self.model_type = 'LeNet5'
        self.layers = []
        self.optim_dictionary = {
            2: ['Conv_Layer_Activation', 4056, []],
            6: ['Conv_Layer_Activation', 1936, []],
            10: ['Classification_Layer_Activation', 120, []],
            12: ['Classification_Layer_Activation', 84, []]
        }

        if settings.OPTIM_MODE == 0:
            for node in range(4056):
                self.optim_dictionary[2][2].append('relu')
            for node in range(1936):
                self.optim_dictionary[6][2].append('relu')
            for node in range(120):
                self.optim_dictionary[10][2].append('relu')
            for node in range(84):
                self.optim_dictionary[12][2].append('relu')
        else:
            self.optim_dictionary[2][2].append('relu')
            self.optim_dictionary[6][2].append('relu')
            self.optim_dictionary[10][2].append('relu')
            self.optim_dictionary[12][2].append('relu')

        channel_size = 1 if settings.DATASET == 0 else 3
        self.layers.append(nn.Conv2d(channel_size, 6, kernel_size=5, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(6))
        self.layers.append(Conv_Layer_Activation(activations=self.optim_dictionary[2][2]))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(16))
        self.layers.append(Conv_Layer_Activation(activations=self.optim_dictionary[6][2]))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(400, 120))
        self.layers.append(Classification_Layer_Activation(activations=self.optim_dictionary[10][2]))
        self.layers.append(nn.Linear(120, 84))
        self.layers.append(Classification_Layer_Activation(activations=self.optim_dictionary[12][2]))
        self.layers.append(nn.Linear(84, num_classes))

        self.net = nn.Sequential(*self.layers)

    def forwards(self, x):
        return self.net(x)


class VGG16(nn.Module):

    #VGG-16 model

    def __init__(self, num_class=10):
        super(VGG16, self).__init__()
        self.model_type = 'VGG16'
        self.layers = []
        self.optim_dictionary = {
            2: ['Conv_Layer_Activation', 65536, []],
            5: ['Conv_Layer_Activation', 65536, []],
            9: ['Conv_Layer_Activation', 32768, []],
            12: ['Conv_Layer_Activation', 32768, []],
            16: ['Conv_Layer_Activation', 16384, []],
            19: ['Conv_Layer_Activation', 16384, []],
            22: ['Conv_Layer_Activation', 16384, []],
            26: ['Conv_Layer_Activation', 8192, []],
            29: ['Conv_Layer_Activation', 8192, []],
            32: ['Conv_Layer_Activation', 8192, []],
            36: ['Conv_Layer_Activation', 2048, []],
            39: ['Conv_Layer_Activation', 2048, []],
            42: ['Conv_Layer_Activation', 2048, []],
            46: ['Classification_Layer_Activation', 4096, []],
            49: ['Classification_Layer_Activation', 4096, []]
        }

        if settings.OPTIM_MODE == 0:
            for node in range(65536):
                self.optim_dictionary[2][2].append('relu')
            for node in range(65536):
                self.optim_dictionary[5][2].append('relu')
            for node in range(32768):
                self.optim_dictionary[9][2].append('relu')
            for node in range(32768):
                self.optim_dictionary[12][2].append('relu')
            for node in range(16384):
                self.optim_dictionary[16][2].append('relu')
            for node in range(16384):
                self.optim_dictionary[19][2].append('relu')
            for node in range(16384):
                self.optim_dictionary[22][2].append('relu')
            for node in range(8192):
                self.optim_dictionary[26][2].append('relu')
            for node in range(8192):
                self.optim_dictionary[29][2].append('relu')
            for node in range(8192):
                self.optim_dictionary[32][2].append('relu')
            for node in range(2048):
                self.optim_dictionary[36][2].append('relu')
            for node in range(2048):
                self.optim_dictionary[39][2].append('relu')
            for node in range(2048):
                self.optim_dictionary[42][2].append('relu')
            for node in range(4096):
                self.optim_dictionary[46][2].append('relu')
            for node in range(4096):
                self.optim_dictionary[49][2].append('relu')

        else:
            self.optim_dictionary[2][2].append('relu')
            self.optim_dictionary[5][2].append('relu')
            self.optim_dictionary[9][2].append('relu')
            self.optim_dictionary[12][2].append('relu')
            self.optim_dictionary[16][2].append('relu')
            self.optim_dictionary[19][2].append('relu')
            self.optim_dictionary[22][2].append('relu')
            self.optim_dictionary[26][2].append('relu')
            self.optim_dictionary[29][2].append('relu')
            self.optim_dictionary[32][2].append('relu')
            self.optim_dictionary[36][2].append('relu')
            self.optim_dictionary[39][2].append('relu')
            self.optim_dictionary[42][2].append('relu')
            self.optim_dictionary[46][2].append('relu')
            self.optim_dictionary[49][2].append('relu')


        self.layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[2][2]))
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[5][2]))
        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[9][2]))
        self.layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[12][2]))
        self.layers.append( nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[16][2]))
        self.layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[19][2]))
        self.layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[22][2]))
        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[26][2]))
        self.layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[29][2]))
        self.layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[32][2]))
        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[36][2]))
        self.layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[39][2]))
        self.layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.layers.append(nn.BatchNorm2d(512))
        self.layers.append(nn.ReLU(inplace=True))#Conv_Layer_Activation(activations=self.optim_dictionary[42][2]))
        self.layers.append(nn.AvgPool2d(kernel_size=2, stride=1))#nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(512, 10))
        # self.layers.append(Classification_Layer_Activation(activations=self.optim_dictionary[46][2]))
        # # self.layers.append(nn.Dropout(p=0.65))
        # self.layers.append(nn.Linear(4096, 4096)) #TODO:
        # self.layers.append(Classification_Layer_Activation(activations=self.optim_dictionary[49][2]))
        # # self.layers.append(nn.Dropout(p=0.65))
        # self.layers.append(nn.Linear(4096, num_class))

        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        return F.softmax(self.net(x), dim=1)
    def forwards(self, x):
        return F.softmax(self.net(x), dim=1)

class ResNet18(nn.Module):

    #ResNet-18 model

    def __init__(self, features, num_class=10):
        super(ResNet18, self).__init__()
        self.model_type = 'ResNet18'
        self.layers = []
        self.optim_dictionary = {}

    def forwards(self, x):
        pass
        #return self.net(x)


#Customized Conv_activation layer
class Conv_Layer_Activation (nn.Module):

    def __init__(self, activations=None):
        super(Conv_Layer_Activation, self).__init__()
        self.layer_type = 'Conv_Layer_Activation'
        self.activations = activations
        self.activation = Activation_Function()
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

        if settings.OPTIM_MODE == 0:
            for j in range(self.activation_size):
                result_temp[:, j] = self.activation(reshape_input[:, j], self.activations[j])
        else:
            result_temp = self.activation(reshape_input, self.activations[0])

        result = result_temp.view(x.size())
        return result


#Customized classification activation layer
class Classification_Layer_Activation(nn.Module):

    def __init__(self,  activations=None):
        super(Classification_Layer_Activation, self).__init__()
        self.layer_type = 'Classification_Layer_Activation'
        self.activations = activations
        self.activation = Activation_Function()
        self.activation_size = None

    def forward(self, x):
        x_size = x.size()
        result = torch.zeros(x_size)
        self.activation_size = x_size [1]
        if torch.cuda.is_available() and settings.GPU_ENABLED:
            result = result.cuda()
        if settings.OPTIM_MODE == 0:
            for i in range(self.activation_size):
                result[:, i] = self.activation(x[:, i], self.activations[i])
        else:
            result = self.activation(x, self.activations[0])
        return result

from torchsummary import summary

def test():
    device = torch.device("cuda")
    net = VGG16()
    net.to(device)
    x = torch.randn(1,3,32,32)
    x=x.to(device)
    y = net.forward(x)
    print(y.size())
    summary(net, input_size=(3, 32, 32), device='cuda')

if __name__ == '__main__':
    test()