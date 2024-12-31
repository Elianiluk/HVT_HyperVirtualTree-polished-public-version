from utils.common_tools import set_random_seed, weights_init_kaiming
import functools

import pandas as pd
import torch.nn as nn
import torch.optim.lr_scheduler
from torchvision.models import vgg11, resnet18

from scripts.prepare_results_to_local import prepare_output_to_local
from data.transforms import BasicTransforms, ImageNetTransforms
from models.network_in_network import NetworkInNetwork
from models.wide_resnet import wide_resnet28_10
from trainers.basic_trainer import BasicTrainer
from utils.constants import INPUT_SIZE, CLASSES_NAMES
from utils.early_stopping import EarlyStopping


class NetworkInNetworkTrainer(BasicTrainer):

    def init_early_stopping(self):
        early_stopping_params = self.early_stopping_params
        if early_stopping_params is None:
            early_stopping_params = {'mode': 'min', 'patience': 50, 'verbose': True}
        return EarlyStopping(**early_stopping_params)

    def _init_model(self):
        num_in_channels = INPUT_SIZE[self.dataset_name][0]
        if self.dataset_name == "CIFAR100":
            model = NetworkInNetwork(num_classes=self.num_classes, num_in_channels=num_in_channels,
                                     config=[(192, 5), (160, 1), (96, 1), 'M', 'D', (192, 5), (192, 1), (192, 1), 'A', 'D', (192, 3), (192, 1), (100, 1)])
        else:
            model = NetworkInNetwork(num_classes=self.num_classes, num_in_channels=num_in_channels,
                                     config=[(192, 5), (160, 1), (96, 1), 'M', 'D', (192, 5), (192, 1), (192, 1), 'A',
                                             'D', (192, 3), (192, 1), (10, 1)])

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.normal_(0, 0.0)
        return model


class ResNetTrainer(BasicTrainer):

    def init_optimizer(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        return optimizer

    def init_transforms(self, padding_mode='constant'):
        if self.dataset_name == 'ImageNet':
            return ImageNetTransforms(self.augment, color_jitter=True)
        return BasicTransforms(self.dataset_name, self.augment, padding_mode=padding_mode)

    def _init_model(self):
        model = resnet18()
        return model


class WideResNetTrainer(BasicTrainer):

    def init_transforms(self, padding_mode='constant'):
        return BasicTransforms(self.dataset_name, self.augment, padding_mode='reflect')

    def init_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [60, 120, 160], gamma=0.2, verbose=True)

    def lr_scheduler_step(self, epoch=-1, train_acc=None, train_loss=None, test_acc=None, test_loss=None):
        self.lr_scheduler.step()

    def init_optimizer(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                    momentum=0.9, weight_decay=5e-4, nesterov=True)
        return optimizer

    def _init_model(self):
        model = wide_resnet28_10(num_classes=self.num_classes,dataset=self.dataset_name)
        return model

    def register_hooks(self, activation_dict):
        def get_activation(activations_dict):
            def hook(model, input, output):
                predictions = output.detach().argmax(dim=1)
                activations_dict['predictions'] = torch.cat([activations_dict['predictions'], predictions])

            return hook

        hook_handles = []
        for name, layer in self.model.named_modules():
            if name == '':  # Assuming the main model module is to be hooked
                activation_dict['predictions'] = torch.Tensor([]).to(self.device)
                handle = layer.register_forward_hook(get_activation(activation_dict))
                hook_handles.append(handle)
        return hook_handles

    def evaluate(self):
        activations_dict = {}
        self.register_hooks(activations_dict)
        super().evaluate()

        metrics_df = pd.DataFrame(columns=['Class', 'Accuracy'])

        targets = torch.tensor(self.datasets.test_set.targets).to(self.device)
        predictions = activations_dict['predictions']

        cls_acc = torch.sum(predictions == targets) / targets.size(0)
        print(f"Class accuracy: {cls_acc * 100:.2f}%")

        # Visualizing class-wise accuracy
        num_images = 0
        for cls in self.classes_indices:
            class_name = CLASSES_NAMES[self.dataset_name][cls]
            print(f"Class: {class_name}")

            cls_idx = torch.where(targets == cls)[0]
            cls_acc = torch.sum(predictions[cls_idx] == cls) / cls_idx.size(0)
            print(f"Accuracy: {cls_acc * 100:.2f}%")

            # Append class accuracy to the DataFrame
            new_row = pd.DataFrame([{'Class': class_name, 'Accuracy': cls_acc.item() * 100}])
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

            results = torch.sum(predictions[cls_idx] == cls).item()

        return metrics_df


class VGG11Trainer(BasicTrainer):

    def init_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [60, 120, 160], gamma=0.2, verbose=True)

    def lr_scheduler_step(self, epoch=-1, train_acc=None, train_loss=None, test_acc=None, test_loss=None):
        self.lr_scheduler.step()

    def _init_model(self):
        model = vgg11(num_classes=self.num_classes, pretrained=True)
        # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        return model


if __name__ == '__main__':
    trainer = NetworkInNetworkTrainer()
    # trainer = WideResNetTrainer()
    print(trainer.dataset_name)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    # trainer = WideResNetTrainer()
    params_num = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f'Number Of Parameters: {params_num}')

    trainer.train_model()
    results = trainer.evaluate()
    # experiment_name = f"{trainer.dataset_name}_{trainer.config['exp_name']}_params_num_{params_num}"
    # prepare_output_to_local(results, experiment_name)
