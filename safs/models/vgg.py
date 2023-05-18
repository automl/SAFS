'''VGG11/13/16/19 in Pytorch.'''
import torch
import utility
import settings
import torch.nn as nn
from torchsummary import summary
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

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
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
        layers_indexes = [2,5,9,12,16,19,22,26,29,32,36,39,42,46,49]
        self.features = self._make_layers(layers_indexes, cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.name = vgg_name


    def forwards(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    def _make_layers(self, layers_indexes, cfg):
        layers = []
        in_channels = 3
        index = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           Conv_Layer_Activation(activations=self.optim_dictionary[layers_indexes[index]][2])]
                in_channels = x
                index = index+1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    # device = torch.device("cpu")
    device = torch.device('cpu')
    net = VGG('VGG16')
    net = net.to(device)
    # for activ in ['relu6','hardswish',
    #                                   'swish','TanhSoft-1','acon',
    #                                   'sin', 'cos', 'gelu','srs', 'linear', 'relu', 'elu', 'selu',
    #                                   'tanh', 'hardtanh', 'softplus',
    #                                   'sigmoid', 'logsigmod']:
    #     for item in net.optim_dictionary:
    #         for i,j in enumerate(net.optim_dictionary[item][2]):
    #             net.optim_dictionary[item][2][i] = activ
    for i,item in enumerate(net.optim_dictionary):
        net.optim_dictionary[item][2][0] = "SReLU"
    x = torch.randn(64,3,32,32)
    x = x.to(device)
    y = net(x)
    print(y.size())
    print(net.name)
    summary(net, input_size=(3, 32, 32), device='cpu')

if __name__ == '__main__':
    test()

