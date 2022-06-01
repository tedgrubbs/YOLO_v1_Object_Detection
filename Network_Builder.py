
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable, Any
from math import ceil, floor

class Network_Builder(torch.nn.Module):
    def __init__(self, config):
        super(Network_Builder, self).__init__()

        self.cfg = config
        img_channels = self.cfg['input_shape'][1]
        img_height = self.cfg['input_shape'][2]
        img_width = self.cfg['input_shape'][3]

        layers = self.cfg['main_net']

        global_activation = self.cfg['global_activation']
        if global_activation == 'gelu':
            global_activation = nn.GELU()
        elif global_activation == 'relu':
            global_activation = nn.ReLU()

        # stores the actual network layers for pytorch to use later
        self.network = nn.ModuleList()

        # records data regarding each layer to build subsequent layers
        network_description = []
        network_description.append({})
        network_description[0]['input_size'] = self.cfg['input_shape'][1:]

        for i in range(len(layers)):

            last_layer_size = network_description[i]['input_size']

            if layers[i]['type'] == "relu_cnnblock":

                new_layer = Relu_CNN_Block(in_channels=last_layer_size[0], out_channels=layers[i]['channels'], kernel_size=layers[i]['kernel_size'], stride=layers[i]['stride'], padding=layers[i]['padding'], groups=1, activation=global_activation)
                example_layer = new_layer.conv
                output_height = floor(((last_layer_size[1] + 2*example_layer.padding[0] - example_layer.dilation[0]*(example_layer.kernel_size[0] - 1) - 1) / example_layer.stride[0]) + 1)
                output_width = floor(((last_layer_size[2] + 2*example_layer.padding[1] - example_layer.dilation[1]*(example_layer.kernel_size[1] - 1) - 1) / example_layer.stride[1]) + 1)
                last_layer_size = [layers[i]['channels'], output_height, output_width]

                print('Layer', i,': CNN block with', layers[i]['channels'], 'channels.', 'Image height:', output_height, 'Image width:', output_width, "Kernel size:", layers[i]['kernel_size'], "Groups:", 1)

            elif layers[i]['type'] == 'maxpool2d':

                new_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=bool(1))
                output_height = ((last_layer_size[1] + 2*new_layer.padding - new_layer.dilation*(new_layer.kernel_size - 1) - 1) / new_layer.stride) + 1
                output_width = ((last_layer_size[2] + 2*new_layer.padding - new_layer.dilation*(new_layer.kernel_size - 1) - 1) / new_layer.stride) + 1
                output_height = ceil(output_height)
                output_width = ceil(output_width)
                last_layer_size = [last_layer_size[0], output_height, output_width]

                print('Layer', i,': MaxPool2d with', last_layer_size[0], 'channels.', 'Image height:', output_height, 'Image width:', output_width)

            elif layers[i]['type'] == 'separable':

                new_layer = Separable_Conv(in_channels=last_layer_size[0], ch1x1=layers[i]['ch1x1'], ch3x3=layers[i]['ch3x3'], stride=layers[i]['stride'], activation=global_activation)
                example_layer = new_layer.conv3x3.conv
                output_height = floor(((last_layer_size[1] + 2*example_layer.padding[0] - example_layer.dilation[0]*(example_layer.kernel_size[0] - 1) - 1) / example_layer.stride[0]) + 1)
                output_width = floor(((last_layer_size[2] + 2*example_layer.padding[1] - example_layer.dilation[1]*(example_layer.kernel_size[1] - 1) - 1) / example_layer.stride[1]) + 1)
                last_layer_size = [layers[i]['ch1x1'], output_height, output_width]

                print('Layer', i,': Separable with', last_layer_size[0], 'channels.', 'Image height:', last_layer_size[1], 'Image width:', last_layer_size[2])

            elif layers[i]['type'] ==  'avgpool2d':

                new_layer = nn.AvgPool2d(kernel_size=(last_layer_size[1],last_layer_size[2]) if layers[i]['kernel_size'] == 'adapt' else layers[i]['kernel_size'], stride=layers[i]['stride'], padding=layers[i]['padding'], ceil_mode=bool(layers[i]['ceil_mode']))
                output_height = ((last_layer_size[1] + 2*new_layer.padding - new_layer.kernel_size[0]) / new_layer.stride) + 1
                output_width = ((last_layer_size[2] + 2*new_layer.padding - new_layer.kernel_size[1]) / new_layer.stride) + 1

                if not layers[i]['ceil_mode']:
                    output_height = floor(output_height)
                    output_width = floor(output_width)
                else:
                    output_height = ceil(output_height)
                    output_width = ceil(output_width)

                last_layer_size = [last_layer_size[0], output_height, output_width]

                print('Layer', i,': AvgPool2d with', last_layer_size[0], 'channels.', 'Image height:', output_height, 'Image width:', output_width)

            elif layers[i]['type'] ==  'linear':

                if len(last_layer_size) > 1:
                    new_output_size = last_layer_size[0]
                    for dim in last_layer_size[1:]:
                        new_output_size *= last_layer_size[dim]
                    last_layer_size = [new_output_size]

                if layers[i]['size'] == 'adapt':
                    new_layer = nn.Linear(last_layer_size[0], last_layer_size[0], bias=layers[i]['bias'])
                else:
                    new_layer = nn.Linear(last_layer_size[0], layers[i]['size'], bias=layers[i]['bias'])

                last_layer_size = [new_layer.out_features]

                print('Layer', i,': linear with size', last_layer_size)

            else :
                print('\nUnsupported layer type:', layers[i]['type'])
                exit()

            self.network.append(new_layer)
            network_description.append({})
            network_description[i+1]['input_size'] = last_layer_size

            if "activation" in layers[i].keys():
                if layers[i]["activation"] == "relu":
                    self.network.append(nn.ReLU())
                    print('Activation: Relu')
                elif layers[i]["activation"] == "elu":
                    self.network.append(nn.ELU())
                    print('Activation: ELU')
                elif layers[i]["activation"] == "gelu":
                    self.network.append(nn.GELU())
                    print('Activation: GELU')
                elif layers[i]["activation"] == "tanh":
                    self.network.append(nn.Tanh())
                    print('Activation: Tanh')
                elif layers[i]["activation"] == "softmax":
                    self.network.append(nn.Softmax(dim=1))
                    print('Activation: Softmax')
                elif layers[i]["activation"] == "sigmoid":
                    self.network.append(nn.Sigmoid())
                    print('Activation: Sigmoid')
                elif layers[i]["activation"] != "default/none":
                    raise NotImplementedError("\nUnsupported activation type:", layers[i]["activation"])
                else:
                    print('Activation: Default or None')

    def forward(self, x):

        for i,layer in enumerate(self.network, 0):

            if isinstance(layer, nn.Linear): x = x.view(-1, layer.in_features)

            x = layer(x)

        return x

# Same as CNN_Block with a relu for convenience
class Relu_CNN_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any):

        super(Relu_CNN_Block, self).__init__()

        self.act = kwargs['activation']
        del kwargs['activation']

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class Separable_Conv(nn.Module):
    def __init__(self, in_channels: int, ch1x1: int, ch3x3: int, stride: int, **kwargs: Any):
        super(Separable_Conv, self).__init__()

        self.conv3x3 = CNN_Block(in_channels, ch3x3, kernel_size=3, padding=1, stride=stride, groups=in_channels)
        self.conv1x1 = CNN_Block(ch3x3, ch1x1, kernel_size=1)
        self.act = kwargs['activation']

    def forward(self, x):
        out = self.conv3x3(x)
        out = self.conv1x1(out)
        return self.act(out)

# Basic cnn with batch norm. No activation function
class CNN_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
