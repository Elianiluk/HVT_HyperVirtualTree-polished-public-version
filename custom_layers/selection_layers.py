import random
import torch
import torch.nn.functional as F
from torch import nn as nn


def saturated_sigmoid(x: torch.Tensor):
    """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
    y = torch.sigmoid(x)
    # noinspection PyTypeChecker
    saturated_sigmoid_ = torch.clamp(1.2 * y - 0.1, 0.0, 1.0)
    return saturated_sigmoid_

class BinarizationLayer(nn.Module):

    def __init__(self, noise_mean=0.0, noise_stddev=1.0):
        super().__init__()
        self.noise_mean = noise_mean
        self.noise_stddev = noise_stddev

    def forward(self, sigma, binarize=None):
        noise = 0
        if self.training:
            noise = torch.normal(self.noise_mean, self.noise_stddev, sigma.size(), device=sigma.device)
        sigma_noised = sigma + noise
        binary_vals = (sigma_noised > 0.0).float()  # gb
        if self.training:  # train with real values
            sat_sigma = saturated_sigmoid(sigma_noised)  # ga
            if binarize is None:
                binarize = random.random() > 0.5
            if binarize:  # train with binary values
                x = binary_vals + sat_sigma - sat_sigma.detach()
            else:
                x = sat_sigma
        else:  # pass only binary values
            sat_sigma = saturated_sigmoid(sigma_noised)
            x = binary_vals
        return x, sat_sigma

class SelectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, noise_mean=0.0, noise_stddev=1.0, reduction_rate=2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        mid_out_channels = max(out_channels // reduction_rate, 1)
        self.fc1 = nn.Linear(in_channels, mid_out_channels)
        self.bn = nn.BatchNorm1d(mid_out_channels)
        self.fc2 = nn.Linear(mid_out_channels, out_channels)
        self.binarization = BinarizationLayer(noise_mean, noise_stddev)

    def forward(self, x, binarize=None):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(self.bn(x))
        sigma = self.fc2(x)
        x = self.binarization(sigma, binarize=binarize)
        return x

class EmbeddingBinarySelectionLayer(SelectionLayer):

    def __init__(self, in_channels, noise_mean=0.0, noise_stddev=1.0, reduction_rate=2, do_batchnorm=False):
        super().__init__(in_channels, 1, noise_mean, noise_stddev, reduction_rate)
        delattr(self, 'fc1')
        delattr(self, 'fc2')
        self.do_batchnorm = do_batchnorm
        if not do_batchnorm:
            delattr(self, 'bn')
        self.fc = nn.Linear(in_channels, 3)
        self.binarization = BinarizationLayer(noise_mean, noise_stddev)
        self.register_buffer('ONES', torch.ones(2))

    def forward(self, x, binarize=None):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        if self.do_batchnorm:
            x = self.bn(x)
        sigmas = self.fc(x)
        if self.training:
            x_b, x_b_r = self.binarization(sigmas[:, 2], binarize=True)
            x, _ = self.binarization(sigmas[:, 0:2], binarize=False)
            return x, x_b, x_b_r
        else:
            x_b, x_b_r = self.binarization(sigmas[:, 2])
            _, sat_sigma = self.binarization(sigmas[:, 0:2])
            return sat_sigma, x_b, x_b_r

class BinarySelectionLayer(SelectionLayer):

    def __init__(self, in_channels, noise_mean=0.0, noise_stddev=1.0, reduction_rate=2, do_batchnorm=False):
        super().__init__(in_channels, 1, noise_mean, noise_stddev, reduction_rate)
        delattr(self, 'fc1')
        delattr(self, 'fc2')
        self.do_batchnorm = do_batchnorm
        if not do_batchnorm:
            delattr(self, 'bn')
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x, binarize=None):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        if self.do_batchnorm:
            x = self.bn(x)
        sigma = self.fc(x)
        x = self.binarization(sigma, binarize=binarize)
        return x

