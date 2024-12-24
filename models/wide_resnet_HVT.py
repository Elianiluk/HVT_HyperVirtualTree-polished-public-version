import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from custom_layers.selection_layers import EmbeddingBinarySelectionLayer

ROOT_URL = 'https://github.com/yoshitomo-matsubara/torchdistill/releases/download'
MODEL_URL_DICT = {
    'cifar10-wide_resnet40_4': ROOT_URL + '/v0.1.1/cifar10-wide_resnet40_4.pt',
    'cifar10-wide_resnet28_10': ROOT_URL + '/v0.1.1/cifar10-wide_resnet28_10.pt',
    'cifar10-wide_resnet16_8': ROOT_URL + '/v0.1.1/cifar10-wide_resnet16_8.pt',
    'cifar100-wide_resnet40_4': ROOT_URL + '/v0.1.1/cifar100-wide_resnet40_4.pt',
    'cifar100-wide_resnet28_10': ROOT_URL + '/v0.1.1/cifar100-wide_resnet28_10.pt',
    'cifar100-wide_resnet16_8': ROOT_URL + '/v0.1.1/cifar100-wide_resnet16_8.pt'
}

INITIAL_SIGMA = 0.5
SCALE_FACTOR = 64

class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideBasicBlockV2(WideBasicBlock):
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideHyperBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate, stride=1, hyper_depth=1):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.hyperconv1 = nn.Linear(hyper_depth, in_planes*out_planes*3*3)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.hyperconv2 = nn.Linear(hyper_depth, out_planes * out_planes * 3 * 3)
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

        self.register_buffer('INITIAL_SIGMA', torch.tensor([INITIAL_SIGMA]))
        self.register_buffer('SCALE_FACTOR', torch.tensor([SCALE_FACTOR]))

    def forward(self, x, h_in=None, binary=None):
        out = self.bn1(x)
        out = self.relu(out)
        if self.training:
            if binary:
                h_in = ((h_in > 1e-4) * self.INITIAL_SIGMA + h_in / self.SCALE_FACTOR)
            else:
                h_in = (self.INITIAL_SIGMA + h_in / self.SCALE_FACTOR)
        else:
            h_in = ((h_in > 1e-4) * self.INITIAL_SIGMA + h_in / self.SCALE_FACTOR)
        outputs = []
        tot_weights = self.hyperconv1(h_in)
        for i, sigma in enumerate(h_in):
            curr_tot_weights = tot_weights[i]
            current_image = out[i:i + 1]
            current_weights = torch.reshape(
                curr_tot_weights,
                (self.out_planes, self.in_planes, 3, 3))
            output = F.conv2d(current_image, current_weights, stride=self.stride, padding=1, bias=None)
            outputs.append(output)
        out = torch.cat(outputs, dim=0)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        outputs = []
        tot_weights = self.hyperconv2(h_in)
        for i, sigma in enumerate(h_in):
            curr_tot_weights = tot_weights[i]
            current_image = out[i:i + 1]
            current_weights = torch.reshape(
                curr_tot_weights,
                (self.out_planes, self.out_planes, 3, 3))
            output = F.conv2d(current_image, current_weights, stride=1, padding=1, bias=None)
            outputs.append(output)
        out = torch.cat(outputs, dim=0)
        out += self.shortcut(x)
        return out

class WideResNet_HyperDecisioNet_1_split(nn.Module):
    def __init__(self, depth, k, dropout_p, num_classes, num_in_channels, stage_sizes=None, norm_layer=None):
        super().__init__()
        n = (depth - 4) / 6
        if stage_sizes is None:
            stage_sizes = [16, 16 * k, 24 * k, 48 * k]
        else:
            stage_sizes = [16, stage_sizes[0] * k, stage_sizes[1] * k, stage_sizes[2] * k]
        in_planes = 16
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(num_in_channels, stage_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.make_wide_layer(in_planes, stage_sizes[1], n, dropout_p, 1)
        self.layer2 = self.make_hyper_wide_layer(stage_sizes[1], stage_sizes[2], n, dropout_p, 2)
        self.layer3 = self.make_wide_layer(stage_sizes[2], stage_sizes[3], n, dropout_p, 2)
        self.bn1 = norm_layer(stage_sizes[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_sizes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.binary_selection_layer1 = EmbeddingBinarySelectionLayer(stage_sizes[1])
        self.binary_selection_layer2 = EmbeddingBinarySelectionLayer(stage_sizes[2])

    def create_sigmas(self, sigmas_b, sigmas_r, binarize):
        if self.training:
            if binarize:
                sigmas0 = (1 - sigmas_b).unsqueeze(1) * sigmas_r[:, 0].unsqueeze(1)
                sigmas1 = sigmas_b.unsqueeze(1) * sigmas_r[:, 1].unsqueeze(1)
            else:
                sigmas0 = sigmas_r[:, 0].unsqueeze(1)
                sigmas1 = sigmas_r[:, 1].unsqueeze(1)
        else:
            sigmas0 = (1 - sigmas_b).unsqueeze(1)
            sigmas1 = sigmas_b.unsqueeze(1)
        return sigmas0, sigmas1
    def _forward_impl(self, x: Tensor, binarize = None) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.layer1(x)
        sigma_r, sigma_b, sigma_b_r = self.binary_selection_layer1(x, binarize)
        # sigma_0, sigma_1 = self.create_sigmas(sigma_b, sigma_r, binarize)
        for i, layer in enumerate(self.layer2):
            if isinstance(layer, WideHyperBasicBlock):
                x0 = layer(x, h_in=sigma_r[:, 0].unsqueeze(1), binary=binarize)
                x1 = layer(x, h_in=sigma_r[:, 1].unsqueeze(1), binary=binarize)
            else:
                x0 = layer(x0)
                x1 = layer(x1)
        sigma_broadcasted = sigma_b[..., None, None, None] if x0.ndim == 4 else sigma_b
        x = (1 - sigma_broadcasted) * x0 + (sigma_broadcasted) * x1
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, sigma_b, sigma_r, sigma_b_r

    def forward(self, x: Tensor, binarize=None) -> Tensor:
        return self._forward_impl(x, binarize)
    @staticmethod
    def make_wide_layer(in_planes: int,
                        out_planes: int,
                        num_blocks: int,
                        dropout_rate: float,
                        stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(WideBasicBlockV2(in_planes, out_planes, dropout_rate, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    @staticmethod
    def make_hyper_wide_layer(in_planes: int,
                        out_planes: int,
                        num_blocks: int,
                        dropout_rate: float,
                        stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        layers.append(WideHyperBasicBlock(in_planes, out_planes, dropout_rate, stride))
        in_planes = out_planes
        for stride in strides[1:]:
            layers.append(WideBasicBlockV2(in_planes, out_planes, dropout_rate, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

class WideResNet_HyperDecisioNet_2_split(nn.Module):
    def __init__(self, depth, k, dropout_p, num_classes, num_in_channels, stage_sizes, norm_layer=None):
        super().__init__()
        n = (depth - 4) / 6
        if stage_sizes is None:
            stage_sizes = [16, 16 * k, 24 * k, 48 * k]
        else:
            stage_sizes = [16, stage_sizes[0] * k, stage_sizes[1] * k, stage_sizes[2] * k]
        in_planes = 16
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(num_in_channels, stage_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.make_wide_layer(in_planes, stage_sizes[1], n, dropout_p, 1)
        self.layer2 = self.make_hyper_wide_layer(stage_sizes[1], stage_sizes[2], n, dropout_p, 2)
        self.layer3 = self.make_hyper_wide_layer(stage_sizes[2], stage_sizes[3], n, dropout_p, 2)
        self.bn1 = norm_layer(stage_sizes[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_sizes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.binary_selection_layer1 = EmbeddingBinarySelectionLayer(stage_sizes[1])
        self.binary_selection_layer2 = EmbeddingBinarySelectionLayer(stage_sizes[2])

    def create_sigmas(self, sigmas_b, sigmas_r, binarize):
        if self.training:
            if binarize:
                sigmas0 = (1 - sigmas_b).unsqueeze(1) * sigmas_r[:, 0].unsqueeze(1)
                sigmas1 = sigmas_b.unsqueeze(1) * sigmas_r[:, 1].unsqueeze(1)
            else:
                sigmas0 = sigmas_r[:, 0].unsqueeze(1)
                sigmas1 = sigmas_r[:, 1].unsqueeze(1)
        else:
            sigmas0 = (1 - sigmas_b).unsqueeze(1)
            sigmas1 = sigmas_b.unsqueeze(1)
        return sigmas0, sigmas1
    def _forward_impl(self, x: Tensor, binarize = None) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.layer1(x)
        sigmas_r_1, sigmas_b_1, sigmas_b_r_1 = self.binary_selection_layer1(x, binarize)
        # sigmas_1_0, sigmas_1_1 = self.create_sigmas(sigmas_b_1, sigmas_r_1, binarize)
        for i, layer in enumerate(self.layer2):
            if isinstance(layer, WideHyperBasicBlock):
                x0 = layer(x, h_in=sigmas_r_1[:, 0].unsqueeze(1), binary=binarize)
                x1 = layer(x, h_in=sigmas_r_1[:, 1].unsqueeze(1), binary=binarize)
            else:
                x0 = layer(x0)
                x1 = layer(x1)
        sigma_broadcasted = sigmas_b_1[..., None, None, None] if x0.ndim == 4 else sigmas_b_1
        x = (1 - sigma_broadcasted) * x0 + (sigma_broadcasted) * x1
        sigmas_r_2, sigmas_b_2, sigmas_b_r_2 = self.binary_selection_layer2(x, binarize)
        # sigmas_2_0, sigmas_2_1 = self.create_sigmas(sigmas_b_2, sigmas_r_2, binarize)
        for i, layer in enumerate(self.layer3):
            if isinstance(layer, WideHyperBasicBlock):
                x0 = layer(x, h_in=sigmas_r_2[:, 0].unsqueeze(1), binary=binarize)
                x1 = layer(x, h_in=sigmas_r_2[:, 0].unsqueeze(1), binary=binarize)
            else:
                x0 = layer(x0)
                x1 = layer(x1)

        sigma_broadcasted = sigmas_b_2[..., None, None, None] if x0.ndim == 4 else sigmas_b_2
        x = (1 - sigma_broadcasted) * x0 + (sigma_broadcasted) * x1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        sigmas_b = torch.column_stack([sigmas_b_1, sigmas_b_2])
        sigmas_r = torch.column_stack([sigmas_r_1, sigmas_r_2])
        if not self.training:
            sigmas_b_r = torch.column_stack([sigmas_b_r_1, sigmas_b_r_2])
            return x, sigmas_b, sigmas_r, sigmas_b_r
        else:
            return x, sigmas_b, sigmas_r, None

    def forward(self, x: Tensor, binarize=None) -> Tensor:
        return self._forward_impl(x, binarize)
    @staticmethod
    def make_wide_layer(in_planes: int,
                        out_planes: int,
                        num_blocks: int,
                        dropout_rate: float,
                        stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(WideBasicBlockV2(in_planes, out_planes, dropout_rate, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    @staticmethod
    def make_hyper_wide_layer(in_planes: int,
                        out_planes: int,
                        num_blocks: int,
                        dropout_rate: float,
                        stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        layers.append(WideHyperBasicBlock(in_planes, out_planes, dropout_rate, stride))
        in_planes = out_planes
        for stride in strides[1:]:
            layers.append(WideBasicBlockV2(in_planes, out_planes, dropout_rate, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)


if __name__ == '__main__':
    from torchinfo import summary

    # wresnet = wide_resnet28_10(num_classes=100)
    wresnet = WideResNet_HyperDecisioNet_2_split(28, 10, dropout_p=0.3, num_classes=10)
    summary(wresnet, (1, 3, 32, 32))
