"""
Author: Yawei Li
Email: yawei.li@vision.ee.ethz.ch
Introducing locality mechanism to "DeiT: Data-efficient Image Transformers".
"""
import torch
import torch.nn as nn
import math
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import DropPath
from timm.models.registry import register_model
import argparse

from timm.models import create_model



#############################################################################

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

        reshape_input = x.reshape(x.size()[0], -1)
        self.activation_size = reshape_input.size()[1]
        result_temp = torch.zeros(reshape_input.size())
        if torch.cuda.is_available():
            result_temp = result_temp.cuda()

        if settings.OPTIM_MODE == 0:#TODO:
            for j in range(self.activation_size):
                result_temp[:, j] = self.activation(reshape_input[:, j], self.activations[j])
        else:
            result_temp = self.activation(reshape_input, self.activations[0])

        result = result_temp.view(x.size())
        return result


#############################################################################







class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid() #TODO:
        else:
            self.sigmoid = h_sigmoid() #TODO:

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True), #TODO:
                nn.Linear(channel // reduction, channel),
                h_sigmoid() #TODO:
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LocalityFeedForward(nn.Module):#TODO: 123 optim_dictionary,num_block,
    def __init__(self,optim_dictionary,num_block, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else Conv_Layer_Activation\
                (activations=optim_dictionary[(num_block,0)][2],\
                name='x', size=10)
            ]) #TODO:

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else Conv_Layer_Activation\
                    (activations=optim_dictionary[((num_block,1))][2] , name='x', size=10) #TODO:
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, qk_reduce=1, attn_drop=0., proj_drop=0.):
        """
        :param dim:
        :param num_heads:
        :param qkv_bias:
        :param qk_scale:
        :param qk_reduce: reduce the output dimension for QK projection
        :param attn_drop:
        :param proj_drop:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_reduce = qk_reduce
        self.dim = dim
        self.qk_dim = int(dim / self.qk_reduce)

        self.qkv = nn.Linear(dim, int(dim * (1 + 1 / qk_reduce * 2)), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.qk_reduce == 1:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            q, k, v = torch.split(self.qkv(x), [self.qk_dim, self.qk_dim, self.dim], dim=-1)
            q = q.reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = k.reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = v.reshape(B, N, self.num_heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):#TODO:235 and 245 optim_dictionary, i, ///
    def __init__(self, optim_dictionary, i, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, qk_reduce=1, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act='hs+se', reduction=4, wo_dp_conv=False, dp_first=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, qk_reduce=qk_reduce,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # The MLP is replaced by the conv layers.
        self.conv = LocalityFeedForward(optim_dictionary, i, dim, dim, 1, mlp_ratio, act, reduction, wo_dp_conv, dp_first)

    def forward(self, x):
        batch_size, num_token, embed_dim = x.shape                                  # (B, 197, dim)
        patch_size = int(math.sqrt(num_token))

        x = x + self.drop_path(self.attn(self.norm1(x)))                            # (B, 197, dim)
        # Split the class token and the image token.
        cls_token, x = torch.split(x, [1, num_token - 1], dim=1)                    # (B, 1, dim), (B, 196, dim)
        # Reshape and update the image token.
        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)   # (B, dim, 14, 14)
        x = self.conv(x).flatten(2).transpose(1, 2)                                 # (B, 196, dim)
        # Concatenate the class token and the newly computed image token.
        x = torch.cat([cls_token, x], dim=1)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        #########################################
        # Origianl implementation
        # self.norm2 = norm_layer(dim)
        #         mlp_hidden_dim = int(dim * mlp_ratio)
        #         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        #########################################

        # Replace the MLP layer by LocalityFeedForward.
        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, act='hs+se', reduction=dim//4)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        #########################################
        # Origianl implementation
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        #########################################

        # Change the computation accordingly in three steps.
        batch_size, num_token, embed_dim = x.shape
        patch_size = int(math.sqrt(num_token))
        # 1. Split the class token and the image token.
        cls_token, x = torch.split(x, [1, embed_dim - 1], dim=1)
        # 2. Reshape and update the image token.
        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)
        x = self.conv(x).flatten(2).transpose(1, 2)
        # 3. Concatenate the class token and the newly computed image token.
        x = torch.cat([cls_token, x], dim=1)
        return x


class LocalVisionTransformer(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=16, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 act=3, reduction=4, wo_dp_conv=False, dp_first=False):
        # print(hybrid_backbone is None)
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                         num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                         drop_path_rate)
#####################################################################################
        self.optim_dictionary = {
            (0, 0): ['Conv_Layer_Activation', 225792, []],
            (0, 1): ['Conv_Layer_Activation', 225792, []],
            (1, 0): ['Conv_Layer_Activation', 225792, []],
            (1, 1): ['Conv_Layer_Activation', 225792, []],
            (2, 0): ['Conv_Layer_Activation', 225792, []],
            (2, 1): ['Conv_Layer_Activation', 225792, []],
            (3, 0): ['Conv_Layer_Activation', 225792, []],
            (3, 1): ['Conv_Layer_Activation', 225792, []],
            (4, 0): ['Conv_Layer_Activation', 225792, []],
            (4, 1): ['Conv_Layer_Activation', 225792, []],
            (5, 0): ['Conv_Layer_Activation', 225792, []],
            (5, 1): ['Conv_Layer_Activation', 225792, []],
            (6, 0): ['Conv_Layer_Activation', 225792, []],
            (6, 1): ['Conv_Layer_Activation', 225792, []],
            (7, 0): ['Conv_Layer_Activation', 225792, []],
            (7, 1): ['Conv_Layer_Activation', 225792, []],
            (8, 0): ['Conv_Layer_Activation', 225792, []],
            (8, 1): ['Conv_Layer_Activation', 225792, []],
            (9, 0): ['Conv_Layer_Activation', 225792, []],
            (9, 1): ['Conv_Layer_Activation', 225792, []],
            (10, 0): ['Conv_Layer_Activation', 225792, []],
            (10, 1): ['Conv_Layer_Activation', 225792, []],
            (11, 0): ['Conv_Layer_Activation', 225792, []],
            (11, 1): ['Conv_Layer_Activation', 225792, []]
        }
        if settings.OPTIM_MODE == 0:
            pass
        else:
            self.optim_dictionary[(0,0)][2].append('relu')
            self.optim_dictionary[(0,1)][2].append('relu')
            self.optim_dictionary[(1,0)][2].append('relu')
            self.optim_dictionary[(1,1)][2].append('relu') 
            self.optim_dictionary[(2,0)][2].append('relu')
            self.optim_dictionary[(2,1)][2].append('relu')
            self.optim_dictionary[(3,0)][2].append('relu')
            self.optim_dictionary[(3,1)][2].append('relu')            
            self.optim_dictionary[(4,0)][2].append('relu')
            self.optim_dictionary[(4,1)][2].append('relu')
            self.optim_dictionary[(5,0)][2].append('relu')
            self.optim_dictionary[(5,1)][2].append('relu')             
            self.optim_dictionary[(6,0)][2].append('relu')
            self.optim_dictionary[(6,1)][2].append('relu')
            self.optim_dictionary[(7,0)][2].append('relu')
            self.optim_dictionary[(7,1)][2].append('relu') 
            self.optim_dictionary[(8,0)][2].append('relu')
            self.optim_dictionary[(8,1)][2].append('relu')
            self.optim_dictionary[(9,0)][2].append('relu')
            self.optim_dictionary[(9,1)][2].append('relu') 
            self.optim_dictionary[(10,0)][2].append('relu')
            self.optim_dictionary[(10,1)][2].append('relu')
            self.optim_dictionary[(11,0)][2].append('relu')
            self.optim_dictionary[(11,1)][2].append('relu') 
#####################################################################################

        # parse act
        if act == 1:
            act = 'relu6'
        elif act == 2:
            act = 'hs'
        elif act == 3:
            act = 'hs+se'
        elif act == 4:
            act = 'hs+eca'
        else:
            act = 'hs+ecah'

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block( optim_dictionary=self.optim_dictionary,i=i,#TODO:
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act=act, reduction=reduction, wo_dp_conv=wo_dp_conv, dp_first=dp_first
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)



@register_model
def localvit_tiny_mlp6_act1(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=6, qkv_bias=True, act=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# reduction = 4
@register_model
def localvit_tiny_mlp4_act3_r4(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, act=3, reduction=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# reduction = 192
@register_model
def localvit_tiny_mlp4_act3_r192(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, act=3, reduction=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def localvit_small_mlp4_act3_r384(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True, act=3, reduction=384,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return 






def test():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # device = torch.device("cpu")
    args.nb_classes = 1000
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    device = torch.device('cpu')

    model.to(device)

    for i,item in enumerate(model.optim_dictionary):
        model.optim_dictionary[item][2][0] = "SReLU"
    x = torch.randn(64,3,224,224)
    x = x.to(device)
    y = model(x)

    #summary(net, input_size=(3, 32, 32), device='cpu')

def create_model_localvit():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # device = torch.device("cpu")
    args.nb_classes = 1000
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    return model

def create_args_localvit(state):
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if state.args.training_enabler == True:
        args.epochs = state.args.train_epochs
    else:
        args.epochs = state.args.optim_epochs
    args.batch_size = state.args.b
    linear_scaled_lr = 5e-4 * state.args.b * state.args.world_size/ 512.0
    args.lr = linear_scaled_lr
    return args

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=2048, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='localvit_tiny_mlp6_act1', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=16, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser