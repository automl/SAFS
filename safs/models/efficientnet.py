'''EfficientNet in PyTorch.

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
'''
import torch
import settings
import torch.nn as nn
import torch.nn.functional as F
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
        # print(x.size(),self.activations,self.name)
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


class Block(nn.Module):
    '''expand + depthwise + pointwise + squeeze-excitation'''

    def __init__(self, optim_dictionary, index, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        # self.conv1 = nn.Conv2d(
        #     in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=stride, padding=1, groups=planes, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(
        #     planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        # SE layers
        self.fc1 = nn.Conv2d(out_planes, out_planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(out_planes//16, out_planes, kernel_size=1)

        #self.shortcut = self._make_shortcut(in_planes, planes, stride)
        self.active = Conv_Layer_Activation(activations=optim_dictionary[
            (index,"active")][2], name='block1', size=10)
        self.features = self.make_block_1(optim_dictionary, index, in_planes, out_planes, planes, stride)

    # def _make_shortcut(self, in_planes, out_planes, stride):
    #     self.shortcut_ = nn.Sequential()
    #     if stride == 1 and in_planes != out_planes:
    #         self.shortcut_ = nn.Sequential(
    #             nn.Conv2d(in_planes, out_planes, kernel_size=1,
    #                       stride=1, padding=0, bias=False),
    #             nn.BatchNorm2d(out_planes),
    #         )
    #     return shortcut_

    def make_block_1(self, optim_dictionary,index, in_planes, out_planes, planes, stride):
        layers = []
        layers += [nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)]
        layers += [nn.BatchNorm2d(planes)]
        layers += [Conv_Layer_Activation(activations=optim_dictionary[(index,"features",2)][2], name='block2', size=10)]
        layers += [nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)]
        layers += [nn.BatchNorm2d(planes)]
        layers += [Conv_Layer_Activation(activations=optim_dictionary[(index,"features",5)][2], name='block3', size=10)]
        layers += [nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)]
        layers += [nn.BatchNorm2d(out_planes)]
        return nn.Sequential(*layers)

    def forwards_(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = self.active(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out
    def forward(self, x):
        out = self.features(x)
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = self.active(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.linear = nn.Linear(cfg[-1][1], num_classes)

        self.optim_dictionary = {
            ("active"): ['Conv_Layer_Activation', 32768, []],
            (0, "active"): ['Conv_Layer_Activation', 32768, []],
            (0,"features",2): ['Conv_Layer_Activation', 8192, []],
            (0,"features",5): ['Conv_Layer_Activation', 1, []],
            (1, "active"): ['Conv_Layer_Activation', 24576, []],
            (1,"features",2): ['Conv_Layer_Activation', 24576, []],
            (1,"features",5): ['Conv_Layer_Activation', 1, []],
            (2, "active"): ['Conv_Layer_Activation', 36864, []],
            (2,"features",2): ['Conv_Layer_Activation', 36864, []],
            (2,"features",5): ['Conv_Layer_Activation', 1, []],
            (3, "active"): ['Conv_Layer_Activation', 36864, []],
            (3,"features",2): ['Conv_Layer_Activation', 9216, []],
            (3,"features",5): ['Conv_Layer_Activation', 2, []],
            (4, "active"): ['Conv_Layer_Activation', 15360, []],
            (4,"features",2): ['Conv_Layer_Activation', 15360, []],
            (4,"features",5): ['Conv_Layer_Activation', 2, []],
            (5, "active"): ['Conv_Layer_Activation', 15360, []],
            (5,"features",2): ['Conv_Layer_Activation', 3840, []],
            (5,"features",5): ['Conv_Layer_Activation', 5, []],
            (6, "active"): ['Conv_Layer_Activation', 7680, []],           
            (6,"features",2): ['Conv_Layer_Activation', 7680, []],
            (6,"features",5): ['Conv_Layer_Activation', 7680, []],
            (7, "active"): ['Conv_Layer_Activation', 5, []],
            (7,"features",2): ['Conv_Layer_Activation', 7680, []],
            (7,"features",5): ['Conv_Layer_Activation', 7680, []],
            (8, "active"): ['Conv_Layer_Activation', 5, []],
            (8,"features",2): ['Conv_Layer_Activation', 7680, []],
            (8,"features",5): ['Conv_Layer_Activation', 7680, []],
            (9, "active"): ['Conv_Layer_Activation', 7, []],            
            (9,"features",2): ['Conv_Layer_Activation', 10752, []],
            (9,"features",5): ['Conv_Layer_Activation', 10752, []],
            (10, "active"): ['Conv_Layer_Activation', 7, []],
            (10,"features",2): ['Conv_Layer_Activation', 10752, []],
            (10,"features",5): ['Conv_Layer_Activation', 10752, []],
            (11, "active"): ['Conv_Layer_Activation', 7, []],
            (11,"features",2): ['Conv_Layer_Activation', 10752, []],
            (11,"features",5): ['Conv_Layer_Activation', 2688, []],
            (12, "active"): ['Conv_Layer_Activation', 12, []],            
            (12,"features",2): ['Conv_Layer_Activation', 4608, []],
            (12,"features",5): ['Conv_Layer_Activation', 4608, []],
            (13, "active"): ['Conv_Layer_Activation', 12, []],
            (13,"features",2): ['Conv_Layer_Activation', 4608, []],
            (13,"features",5): ['Conv_Layer_Activation', 4608, []],
            (14, "active"): ['Conv_Layer_Activation', 12, []],
            (14,"features",2): ['Conv_Layer_Activation', 4608, []],
            (14,"features",5): ['Conv_Layer_Activation', 4608, []],
            (15,"features",2): ['Conv_Layer_Activation', 4608, []],            
            (15,"features",5): ['Conv_Layer_Activation', 1152, []],            
            (15, "active"): ['Conv_Layer_Activation', 20, []],            
        }
        if settings.OPTIM_MODE == 0:
            for node in range(65536):
                self.optim_dictionary[("layer1",0,"features",2)][2].append('relu')
            for node in range(65536):
                self.optim_dictionary[("layer1",0,"shortcut",0)][2].append('relu')
            for node in range(65536):
                self.optim_dictionary[("layer1",1,"features",2)][2].append('relu')
            for node in range(65536):
                self.optim_dictionary[("layer1",1,"shortcut",0)][2].append('relu')
            for node in range(65536):
                self.optim_dictionary[("active")][2].append('relu')
            for node in range(32768):
                self.optim_dictionary[("layer2",0,"features",2)][2].append('relu')
            for node in range(32768):
                self.optim_dictionary[("layer2",0,"shortcut",0)][2].append('relu')
            for node in range(32768):
                self.optim_dictionary[("layer2",1,"features",2)][2].append('relu')
            for node in range(32768):
                self.optim_dictionary[("layer2",1,"shortcut",0)][2].append('relu')
            for node in range(16960):
                self.optim_dictionary[("layer3",0,"features",2)][2].append('relu')
            for node in range(16960):
                self.optim_dictionary[("layer3",0,"shortcut",0)][2].append('relu')
            for node in range(16960):
                self.optim_dictionary[("layer3",1,"features",2)][2].append('relu')
            for node in range(16960):
                self.optim_dictionary[("layer3",1,"shortcut",0)][2].append('relu')
            for node in range(8192):
                self.optim_dictionary[("layer4",0,"features",2)][2].append('relu')
            for node in range(8192):
                self.optim_dictionary[("layer4",0,"shortcut",0)][2].append('relu')
            for node in range(8192):
                self.optim_dictionary[("layer4",1,"features",2)][2].append('relu')
            for node in range(8192):
                self.optim_dictionary[("layer4",1,"shortcut",0)][2].append('relu')
        else:
            self.optim_dictionary[("active")][2].append('swish')
            self.optim_dictionary[(0,"active")][2].append('swish')
            self.optim_dictionary[(1,"active")][2].append('swish')
            self.optim_dictionary[(2,"active")][2].append('swish')
            self.optim_dictionary[(3,"active")][2].append('swish')
            self.optim_dictionary[(4,"active")][2].append('swish')
            self.optim_dictionary[(5,"active")][2].append('swish')
            self.optim_dictionary[(6,"active")][2].append('swish')
            self.optim_dictionary[(7,"active")][2].append('swish')
            self.optim_dictionary[(8,"active")][2].append('swish')
            self.optim_dictionary[(9,"active")][2].append('swish')
            self.optim_dictionary[(10,"active")][2].append('swish')
            self.optim_dictionary[(11,"active")][2].append('swish')
            self.optim_dictionary[(12,"active")][2].append('swish')
            self.optim_dictionary[(13,"active")][2].append('swish')
            self.optim_dictionary[(14,"active")][2].append('swish')
            self.optim_dictionary[(15,"active")][2].append('swish')
            self.optim_dictionary[(0,"features",2)][2].append('swish')
            self.optim_dictionary[(1,"features",2)][2].append('swish')
            self.optim_dictionary[(2,"features",2)][2].append('swish')
            self.optim_dictionary[(3,"features",2)][2].append('swish')
            self.optim_dictionary[(4,"features",2)][2].append('swish')
            self.optim_dictionary[(5,"features",2)][2].append('swish')
            self.optim_dictionary[(6,"features",2)][2].append('swish')
            self.optim_dictionary[(7,"features",2)][2].append('swish')
            self.optim_dictionary[(8,"features",2)][2].append('swish')
            self.optim_dictionary[(9,"features",2)][2].append('swish')
            self.optim_dictionary[(10,"features",2)][2].append('swish')
            self.optim_dictionary[(11,"features",2)][2].append('swish')
            self.optim_dictionary[(12,"features",2)][2].append('swish')
            self.optim_dictionary[(13,"features",2)][2].append('swish')
            self.optim_dictionary[(14,"features",2)][2].append('swish')
            self.optim_dictionary[(15,"features",2)][2].append('swish')
            self.optim_dictionary[(0,"features",5)][2].append('swish')
            self.optim_dictionary[(1,"features",5)][2].append('swish')
            self.optim_dictionary[(2,"features",5)][2].append('swish')
            self.optim_dictionary[(3,"features",5)][2].append('swish')
            self.optim_dictionary[(4,"features",5)][2].append('swish')
            self.optim_dictionary[(5,"features",5)][2].append('swish')
            self.optim_dictionary[(6,"features",5)][2].append('swish')
            self.optim_dictionary[(7,"features",5)][2].append('swish')
            self.optim_dictionary[(8,"features",5)][2].append('swish')
            self.optim_dictionary[(9,"features",5)][2].append('swish')
            self.optim_dictionary[(10,"features",5)][2].append('swish')
            self.optim_dictionary[(11,"features",5)][2].append('swish')
            self.optim_dictionary[(12,"features",5)][2].append('swish')
            self.optim_dictionary[(13,"features",5)][2].append('swish')
            self.optim_dictionary[(14,"features",5)][2].append('swish')
            self.optim_dictionary[(15,"features",5)][2].append('swish')
        self.layers = self._make_layers(in_planes=32)
        self.active = Conv_Layer_Activation(activations=self.optim_dictionary[("active")][2], name='x', size=10) #TODO:



    def _make_layers(self, in_planes):
        layers = []
        index = 0 
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(self.optim_dictionary, index, in_planes, out_planes, expansion, stride))
                index = index+1
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.active(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def EfficientNetB0(num_classes=10):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 2),
           (6,  24, 2, 1),
           (6,  40, 2, 2),
           (6,  80, 3, 2),
           (6, 112, 3, 1),
           (6, 192, 4, 2),
           (6, 320, 1, 2)]
    return EfficientNet(cfg, num_classes)


def test():
    net = EfficientNetB0()
    for i,item in enumerate(net.optim_dictionary):
        net.optim_dictionary[item][2][0] = "selu"

    net.optim_dictionary
    x = torch.randn(2, 3, 32, 32)
    y = net(x)

    print("haha",y.shape)

if __name__ == '__main__':
    test()
    print('Done!')





# [32768,32768,8192,1,24576,24576,1,36864,36864,1,36864,9216,2,15360,15360,2,15360,3840,5,7680,7680,5,
#  7680,7680,5,7680,7680,7,10752,10752,7,10752,10752,7,10752,2688,12,4608,4608,12,4608,4608,12,4608,
#  4608,12,4608,1152,20,10]