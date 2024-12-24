from typing import List, Union, Callable, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers.selection_layers import BinarySelectionLayer, EmbeddingBinarySelectionLayer
from models.network_in_network import NetworkInNetwork
from models.wide_resnet import WideResNet
from utils.binary_tree import Node
from utils.constants import INPUT_SIZE

CIFAR10_INPUT_CHANNELS = INPUT_SIZE['CIFAR10'][0]
CIFAR10_INPUT_SIZE = INPUT_SIZE['CIFAR10'][1]
SIZE_AFTER_TWO_MAXPOOL = 8
INITIAL_SIGMA = 0.5
SCALE_FACTOR = 64

ConfigTuple = Tuple[Union[int, str, Tuple[int, int]], ...]
ConfigList = List[Any]


class SubNet(nn.Module):
    def __init__(self, initial_hin=0, node_code: Tuple[int, ...] = ()):
        super().__init__()
        self.node_code = node_code
        self.features = nn.Sequential(nn.Conv2d(96, 96, 1, padding=0), nn.ReLU(inplace=True),
                                      nn.Conv2d(96, 96, 1, padding=0), nn.ReLU(inplace=True),
                                      nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                      nn.Dropout(0.5, inplace=True))

    def forward(self, x):
        x = self.features(x)
        return x


class HyperNet(nn.Module):

    def __init__(self, hyper_depth=1, initial_hin=0, node_code: Tuple[int, ...] = ()):
        super().__init__()
        self.node_code = node_code

        self.features = nn.Sequential(nn.Linear(hyper_depth, 230496), nn.ReLU(inplace=True),
                                      nn.Conv2d(96, 96, 1, padding=0), nn.ReLU(inplace=True),
                                      nn.Conv2d(96, 96, 1, padding=0), nn.ReLU(inplace=True),
                                      nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                      nn.Dropout(0.5, inplace=True))

        self.register_buffer('INITIAL_SIGMA', torch.tensor([INITIAL_SIGMA]))
        self.register_buffer('SCALE_FACTOR', torch.tensor([SCALE_FACTOR]))

    def forward(self, x, h_in=None, binary=None):
        for j, feature in enumerate(self.features):
            if 'Linear' in feature.__str__():
                if self.training:
                    if binary:
                        h_in = ((h_in > 1e-4) * self.INITIAL_SIGMA + h_in / self.SCALE_FACTOR)
                    else:
                        h_in = (self.INITIAL_SIGMA + h_in / self.SCALE_FACTOR)
                else:
                    h_in = ((h_in > 1e-4) * self.INITIAL_SIGMA + h_in / self.SCALE_FACTOR)
                outputs = []
                tot_weights = feature(h_in)
                for i, sigma in enumerate(h_in):
                    curr_tot_weights = tot_weights[i]
                    current_image = x[i:i + 1]
                    current_weights = torch.reshape(
                        curr_tot_weights[:230400],
                        (96, 96, 5, 5))
                    current_bias = curr_tot_weights[230400:]
                    output = F.conv2d(current_image, current_weights, current_bias, padding=2)
                    outputs.append(output)
                x = torch.cat(outputs, dim=0)
            else:
                x = feature(x)
        return x, None


class SharedNet(nn.Module):

    def __init__(self, input_channels=3):
        super().__init__()

        hypernet_cls = HyperNet
        subnet_cls = SubNet

        self.before_features = nn.Sequential(nn.Conv2d(input_channels, 192, 5, padding=2), nn.ReLU(inplace=True),
                                             nn.Conv2d(192, 160, 1, padding=0), nn.ReLU(inplace=True),
                                             nn.Conv2d(160, 96, 1, padding=0), nn.ReLU(inplace=True),
                                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                             nn.Dropout(0.5))

        self.after_features = nn.Sequential((nn.Conv2d(96, 96, 3, padding=1)), nn.ReLU(inplace=True),
                                            (nn.Conv2d(96, 96, 1, padding=0)), nn.ReLU(inplace=True),
                                            (nn.Conv2d(96, 10, 1, padding=0)))

        self.hyper = hypernet_cls()
        # self.right = subnet_cls(node_code=(0,))
        # self.left = subnet_cls(node_code=(1,))

        self.binary_selection_layer = EmbeddingBinarySelectionLayer(96)
        self.register_buffer('INITIAL_SIGMA', torch.tensor([INITIAL_SIGMA]))
        self.register_buffer('SCALE_FACTOR', torch.tensor([SCALE_FACTOR]))

    def forward(self, x, binarize):
        x = self.before_features(x)
        sigmas_r, sigmas_b, sigmas_b_r = self.binary_selection_layer(x, binarize)
        if self.training:
            if binarize:
                x0, s0 = self.hyper(x, (1 - sigmas_b).unsqueeze(1)*sigmas_r[:, 0].unsqueeze(1), binary=True)
                x1, s1 = self.hyper(x, sigmas_b.unsqueeze(1)*sigmas_r[:, 1].unsqueeze(1), binary=True)
            else:
                x0, s0 = self.hyper(x, sigmas_r[:, 0].unsqueeze(1), binary=False)
                x1, s1 = self.hyper(x, sigmas_r[:, 1].unsqueeze(1), binary=False)
        else:
            x0, s0 = self.hyper(x, (1 - sigmas_b).unsqueeze(1), binary=True)
            x1, s1 = self.hyper(x, sigmas_b.unsqueeze(1), binary=True)

        sigma_broadcasted = sigmas_b[..., None, None, None] if x0.ndim == 4 else sigmas_b
        x = (1 - sigma_broadcasted) * x0 + (sigma_broadcasted) * x1

        # x = x0 + x1
        x = self.after_features(x)
        return x, sigmas_b, sigmas_r, sigmas_b_r


class NIN_HyperDecisioNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        sharednet_cls = SharedNet
        self.hyperdecisionet = sharednet_cls(input_channels)
        self.classifier = nn.AdaptiveAvgPool2d((1, 1))  # original option
        print("NetworkInNetworkDecisioNet init - Using the following config:")
        print(self)

    def forward(self, x, binarize):
        features_out, sigmas_b, sigmas_r, sigmas_b_r = self.hyperdecisionet(x, binarize)
        out = self.classifier(features_out)
        out = torch.flatten(out, 1)
        return out, sigmas_b, sigmas_r, sigmas_b_r


if __name__ == '__main__':
    from torchinfo import summary

    model = NIN_HyperDecisioNet()
    # out, sigmas = model(images)
    summary(model, (1, 3, 32, 32))
