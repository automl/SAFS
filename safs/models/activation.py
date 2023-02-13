import math
import torch
import random
import torch.nn.functional as AFs
import torch.nn as nn
import settings
import numpy as np

cuda0 = torch.device('cuda:0')

class SReLU():
    def __init__(self, t_left=0, a_left=random.uniform(0, 1),t_right=random.uniform(0, 5),a_right=1):
        super(SReLU, self).__init__()
        self.t_left = nn.Parameter(torch.tensor(t_left), requires_grad=True)
        self.a_left = nn.Parameter(torch.tensor(a_left), requires_grad=True)
        self.t_right = nn.Parameter(torch.tensor(t_right), requires_grad=True)
        self.a_right = nn.Parameter(torch.tensor(a_right), requires_grad=True)

    def forward(self, x):
        # ensure the the right part is always to the right of the left
        t_right = (self.t_right)
        t_left = (self.t_left)
        a_left = (self.a_left)
        a_right = (self.a_right)
        y_left = -torch.nn.ReLU()(-x+t_left) * a_left
        mid = (torch.nn.ReLU()(x-t_left))-(torch.nn.ReLU()(x-t_right))
        y_right = torch.nn.ReLU()(x - t_right) * a_right
        return y_left + y_right + mid


class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    # MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(width, max(r, width // r), kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(max(r, width // r))
        self.fc2 = nn.Conv2d(max(r, width // r), width, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.InstanceNorm1d(width)
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))

    def forward(self, x, active):
        beta = torch.sigmoid(self.bn2(self.fc2(self.fc1(x.view([1,len(x),1,1]))).view([1,len(x)])).view([len(x)]))
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x


# simply define a silu function
def srs(input, a, b):
    return torch.div(input, (input/a + torch.exp(-input/b))) 
    # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
class SRS(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
    def forward(self, input ,a ,b):
        return srs(input,a ,b)

def test(input):
    return input# use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
class TEST(nn.Module):
    def __init__(self):
        super(TEST, self).__init__() # init the base class
        self.ggg = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    def forward(self, input):
        return self.ggg*test(input)

def ash(x, k_ash_):
    result = []
    for input in x:
        input1 = input.cpu()
        # xnp = x.detach().numpy()
        input1 = input1.detach().numpy()
        th = np.percentile(input1, int((1 - k_ash_) * 100))
        m = nn.Threshold(th, 0)
        result.append(m(input))
    for i in range(len(x)):
        x[i] = result[i]
    return x

class ASH(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
    def forward(self, input, k_ash_):
        return ash(input, k_ash_)

class Common_Activation_Function(nn.Module):

    def __init__(self, acon_size):
        super(Common_Activation_Function, self).__init__()
        self.acon_size = acon_size
        # initialize alpha and beta
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.acon_pp = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        # set requiresGrad to true!
        if settings.OPTIM_STAGE == 'Stage_2':
            self.alpha.requires_grad = True
            self.beta.requires_grad = True
            self.acon_pp.requires_grad = True
    def hard_sigmoid(self, x):
        x = (0.2 * x) + 0.5
        x = AFs.threshold(-x, -1, -1)
        x = AFs.threshold(-x, 0, 0)
        return x


    def Unary_Operator(self, x, activation):

        if activation == 'linear':
            return self.beta * x
        elif activation == 'SRelu':
            return SReLU().forward(x)
        elif activation == 'meta_acon': #TODO:
            return MetaAconC(self.acon_size)(x)
        elif activation == 'acon':
            return (self.alpha * x - self.beta * x) * torch.sigmoid(self.acon_pp * (self.alpha * x - self.beta * x)) + self.beta * x
        elif activation == 'TanhSoft-1':
            return nn.Tanh()(self.alpha*x) * nn.Softplus()(x)
        elif activation == 'TanhSoft-2':
            return x * nn.Tanh()(self.alpha * torch.exp(self.beta * x))
        elif activation == 'ash': #TODO:test
            return ASH()(x, self.alpha)
        elif activation == 'srs':#TODO:test
            return SRS()(x, self.alpha, self.beta)
        elif activation == 'test':
            return TEST()(x)
        elif activation == 'mish':
            return self.alpha * nn.Mish()(self.beta * x)
        elif activation == 'relu6':
            return self.alpha * nn.ReLU6()(self.beta * x)
        elif activation == 'hardswish':
            return self.alpha * nn.Hardswish()(self.beta * x)
        elif activation == 'elu':
            return self.alpha * AFs.elu(self.beta * x)
        elif activation == 'relu':
            return self.alpha * AFs.relu(self.beta * x)
        elif activation == 'selu':
            return self.alpha * AFs.selu(self.beta * x)
        elif activation == 'tanh':
            return self.alpha * torch.tanh(self.beta * x)
        elif activation == 'sigmoid':
            return self.alpha * torch.sigmoid(self.beta * x)
        elif activation == 'logsigmiod':
            return self.alpha * AFs.logsigmoid(self.beta * x)
        elif activation == 'hardtan':
            return self.alpha * AFs.hardtan(self.beta * x)
        elif activation == 'softplus':
            return self.alpha * AFs.softplus(self.beta * x)
        elif activation == 'swish':
            return self.alpha * torch.sigmoid(self.beta * x) * x
        elif activation == 'sin':
            return self.alpha * torch.sin(self.beta * x)
        elif activation == 'cos':
            return self.alpha * torch.cos(self.beta * x)
        elif activation == 'gelu':
            return self.alpha * nn.GELU()(self.beta * x)
        elif activation == 'elish':
            return self.alpha * torch.where(x < 0, AFs.elu(self.beta * x) * torch.sigmoid(self.beta * x), torch.sigmoid(self.beta * x) * x)
        elif activation == 'hard_elish':
            return self.alpha * torch.where(x < 0.0, AFs.elu(self.beta * x) * self.hard_sigmoid(self.beta * x), self.hard_sigmoid(self.beta * x) * x)
        else:
            return self.alpha * AFs.relu(self.beta * x)
            print('xx')

    def forward(self, x, activation):
        #print (self.alpha.requires_grad)
        #print(self.beta.item())
        return self.Unary_Operator(x, activation)

def Activation_Function(name, x):
    # return MetaAconC(x)
    return Common_Activation_Function(x)