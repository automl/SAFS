'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.activation import Activation_Function
import settings
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

        if settings.OPTIM_MODE == 0:
            for j in range(self.activation_size):
                result_temp[:, j] = self.activation(reshape_input[:, j], self.activations[j])
        else:
            result_temp = self.activation(reshape_input, self.activations[0])

        result = result_temp.view(x.size())
        return result

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,optim_dictionary, layer_name, num_block, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.features= self._make_BasicBlock(optim_dictionary,layer_name, num_block, in_planes, planes, stride)
        self.shortcut = self._make_shortcut(optim_dictionary,layer_name,num_block,in_planes, planes, stride)

    def _make_shortcut(self,optim_dictionary,layer_name,num_block, in_planes, planes, stride):
        shortcut = nn.Sequential(Conv_Layer_Activation(activations=optim_dictionary[
            (layer_name,num_block,"shortcut",0)][2], name='x', size=10))
        if stride != 1 or in_planes != self.expansion*planes:
            shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                Conv_Layer_Activation(activations=optim_dictionary[
                    (layer_name,num_block,"shortcut",0)][2] , name='x', size=10)
            )
        return shortcut
    def _make_BasicBlock(self, optim_dictionary,layer_name, num_block, in_planes, planes, stride):
        layers = []
        layers += [nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(planes)]
        layers += [Conv_Layer_Activation(activations = optim_dictionary
                                         [(layer_name,num_block,"features",2)][2] , name='x', size=10)]
        layers += [nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(planes)]
        #layers += [self._make_shortcut(in_planes, planes, stride)]
        #layers += [Conv_Layer_Activation(activations='relu', name='x', size=10)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        return out
    def forward_(self, x):
        out = self.active(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.active(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.active = Conv_Layer_Activation(activations='relu', name='x', size=10)

    def forward(self, x):
        out = self.active(self.bn1(self.conv1(x)))

        out = self.active(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.active(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.optim_dictionary = {
            ("layer1",0,"features",2): ['Conv_Layer_Activation', 65536, []],
            ("layer1",0,"shortcut",0): ['Conv_Layer_Activation', 65536, []],
            ("layer1",1,"features",2): ['Conv_Layer_Activation', 65536, []],
            ("layer1",1,"shortcut",0): ['Conv_Layer_Activation', 65536, []],
            ("layer2",0,"features",2): ['Conv_Layer_Activation', 32768, []],
            ("layer2",0,"shortcut",0): ['Conv_Layer_Activation', 32768, []],
            ("layer2",1,"features",2): ['Conv_Layer_Activation', 32768, []],
            ("layer2",1,"shortcut",0): ['Conv_Layer_Activation', 32768, []],
            ("layer3",0,"features",2): ['Conv_Layer_Activation', 16960, []],
            ("layer3",0,"shortcut",0): ['Conv_Layer_Activation', 16960, []],
            ("layer3",1,"features",2): ['Conv_Layer_Activation', 16960, []],
            ("layer3",1,"shortcut",0): ['Conv_Layer_Activation', 16960, []],
            ("layer4",0,"features",2): ['Conv_Layer_Activation', 8192, []],
            ("layer4",0,"shortcut",0): ['Conv_Layer_Activation', 8192, []],
            ("layer4",1,"features",2): ['Conv_Layer_Activation', 8192, []],
            ("layer4",1,"shortcut",0): ['Conv_Layer_Activation', 8192, []],
            ("active"): ['Conv_Layer_Activation', 65536, []]
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
            self.optim_dictionary[("layer1",0,"features",2)][2].append('relu')
            self.optim_dictionary[("layer1",0,"shortcut",0)][2].append('relu')
            self.optim_dictionary[("layer1",1,"features",2)][2].append('relu')
            self.optim_dictionary[("layer1",1,"shortcut",0)][2].append('relu')
            self.optim_dictionary[("layer2",0,"features",2)][2].append('relu')
            self.optim_dictionary[("layer2",0,"shortcut",0)][2].append('relu')
            self.optim_dictionary[("layer2",1,"features",2)][2].append('relu')
            self.optim_dictionary[("layer2",1,"shortcut",0)][2].append('relu')
            self.optim_dictionary[("layer3",0,"features",2)][2].append('relu')
            self.optim_dictionary[("layer3",0,"shortcut",0)][2].append('relu')
            self.optim_dictionary[("layer3",1,"features",2)][2].append('relu')
            self.optim_dictionary[("layer3",1,"shortcut",0)][2].append('relu')
            self.optim_dictionary[("layer4",0,"features",2)][2].append('relu')
            self.optim_dictionary[("layer4",0,"shortcut",0)][2].append('relu')
            self.optim_dictionary[("layer4",1,"features",2)][2].append('relu')
            self.optim_dictionary[("layer4",1,"shortcut",0)][2].append('relu')
            self.optim_dictionary[("active")][2].append('relu')
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer("layer1", block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer("layer2", block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer("layer3", block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer("layer4", block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.active = Conv_Layer_Activation(activations=self.optim_dictionary[("active")][2], \
            name='x', size=10) #TODO:

        if block == BasicBlock:
            self.name = "resnet" + str(sum(num_blocks) * 2 + 2)
        else:
            self.name = "resnet" + str(sum(num_blocks) * 3 + 2)

    def _make_layer(self, layer_name, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for num_block,stride in enumerate(strides):
            layers.append(block(self.optim_dictionary ,layer_name, num_block, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.active(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def forwards(self, x):
        out = self.active(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    print(net.name)
    summary(net, input_size=(3, 32, 32), device='cpu')


if __name__ == '__main__':
    test()
