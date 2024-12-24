import random
import functools
import torch
import torch.nn as nn
import wandb
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from models import wide_resnet
from trainers.basic_trainer import BasicTrainer
from scripts.prepare_results_to_local import prepare_output_to_local
# from custom_layers.losses import WeightedMSELoss
from data.datasets import FilteredRelabeledDatasets
from models.nin_HVT import NIN_HyperDecisioNet
from utils.constants import LABELS_MAP, CLASSES_NAMES, INPUT_SIZE, NUM_CLASSES
from utils.metrics_tracker import SigmaLossMetricsTracker
from utils.common_tools import set_random_seed, weights_init_kaiming
import torch.nn.init as init
from models.wide_resnet_HVT import WideResNet_HyperDecisioNet_2_split, WideResNet_HyperDecisioNet_1_split

WRESNET_STAGE_SIZES = {'100_baseline': ([16, 16, 16], 3),
                       '100_baseline_single_early': ([16, 16, 32], 2)}

class HVT_Trainer(BasicTrainer):

    def __init__(self):
        super().__init__()
        # sigma_weights = self._init_sigma_weights()
        self.sigma_criterion = nn.MSELoss()  # WeightedMSELoss(sigma_weights)
        self.metrics_tracker = SigmaLossMetricsTracker(self.include_top5, self.include_entropy)

    def _init_model(self):
        raise NotImplementedError

    def _init_config_attributes(self):
        super()._init_config_attributes()
        self.beta = self.config['beta']
        self.always_binarize = self.config['always_binarize']
        self.include_entropy = self.config['include_entropy']

    def init_data_sets(self):
        labels_map = dict(LABELS_MAP[self.dataset_name])
        return FilteredRelabeledDatasets(self.transforms, use_validation=self.use_validation,
                                         classes_indices=self.classes_indices,
                                         labels_map=labels_map,
                                         dataset_name=self.dataset_name)

    def _feed_forward(self, inputs, targets):
        cls_targets, *sigma_targets = targets
        sigma_targets = torch.column_stack(sigma_targets)
        binarize = self.always_binarize or random.random() > 0.5
        outputs, sigma_b, sigma_r, _ = self.model(inputs, binarize=binarize)
        # print(f'Cls correct: {sum(torch.argmax(outputs, 1)==cls_targets)}')
        # print(f'Sigma correct: {sum(sigmas==sigma_targets)}\n')
        cls_loss = self.cls_criterion(outputs, cls_targets.long())
        sigma_loss = self.sigma_criterion(sigma_b.unsqueeze(1), sigma_targets.float())
        combined_loss = cls_loss + self.beta * sigma_loss
        self.metrics_tracker.update(cls_loss, sigma_loss, combined_loss, outputs, cls_targets, sigma_b.unsqueeze(1), sigma_targets)
        return outputs, combined_loss

    def _single_epoch(self, epoch: int, train_test_val: str):
        norm_acc, norm_loss = super()._single_epoch(epoch, train_test_val)
        if (train_test_val == 'test') and self.include_entropy:
            print(f'Average Leaf Entrophy: {self.metrics_tracker.get_leaf_entrophy()}')
        if self.use_wandb:
            log_dict = {f"{train_test_val}_cls_loss": self.metrics_tracker.get_norm_cls_loss(),
                        f"{train_test_val}_sigma_loss": self.metrics_tracker.get_norm_sigma_loss(),
                        f"{train_test_val}_sigma_accuracy": 100. * self.metrics_tracker.get_norm_sigma_acc()}
            wandb.log(log_dict, step=epoch + 1)
        return norm_acc, norm_loss

    def input_and_targets_to_device(self, inputs, targets):
        inputs = inputs.to(self.device)
        for i in range(len(targets)):
            targets[i] = targets[i].to(self.device)
        return inputs, targets

    def init_parser(self):
        parser = super().init_parser()
        parser.add_argument('--beta', type=float, help='weight for the sigma loss', default=0.0)
        parser.add_argument('--always_binarize', action='store_true', help='do not use non-binary values in the binarization layer (i.e., perform only hard routing)')
        parser.add_argument('--include_entropy', action='store_true', help="Whether to do entrophy calculation")
        return parser

    def register_hooks(self, activation_dict):
        def get_activation(activations_dict):
            def hook(model, input, output):
                predictions = output[0].detach()
                predictions = predictions.argmax(dim=1)
                sigma = output[1].detach()
                sigma_r = output[2].detach()
                sigmas_b_r = output[3].detach()
                activations_dict['predictions'] = torch.cat([activations_dict['predictions'], predictions])
                activations_dict['sigma'] = torch.cat([activations_dict['sigma'], sigma])
                activations_dict['sigma_r'] = torch.cat([activations_dict['sigma_r'], sigma_r])
                activations_dict['sigmas_b_r'] = torch.cat([activations_dict['sigmas_b_r'], sigmas_b_r])

            return hook

        hook_handles = []
        for name, layer in self.model.named_modules():
            if name == '':
                activation_dict['predictions'] = torch.Tensor([]).to(self.device)
                activation_dict['sigma'] = torch.Tensor([]).to(self.device)
                activation_dict['sigma_r'] = torch.Tensor([]).to(self.device)
                activation_dict['sigmas_b_r'] = torch.Tensor([]).to(self.device)
                handle = layer.register_forward_hook(get_activation(activation_dict))
                hook_handles.append(handle)
        return hook_handles

    def evaluate(self):
        activations_dict = {}
        self.register_hooks(activations_dict)
        super().evaluate()

        targets = torch.tensor(self.datasets.test_set.targets).to(self.device)
        predictions = activations_dict['predictions']
        sigmas = activations_dict['sigma']
        sigmas_r = activations_dict['sigma_r']
        sigmas_b_r = activations_dict['sigmas_b_r']

        cls_targets = targets[:, 0]
        sigma_targets = targets[:, 1:]

        metrics_df = pd.DataFrame(columns=['Scope', 'Class', 'Accuracy', 'Sigma Accuracy', 'Amount'])

        sigmas = sigmas.unsqueeze(1) if sigmas.dim() == 1 else sigmas
        sigmas_b_r = sigmas_b_r.unsqueeze(1) if sigmas_b_r.dim() == 1 else sigmas_b_r

        # Entire Model Analysis
        print("Entire Model Analysis: ")
        cls_acc = (predictions == cls_targets).sum().item() / targets.size(0)
        sigma_diffs = (sigmas == sigma_targets)
        encoding = {0: 'both wrong', 1: 'first correct', 2: 'second correct', 3: 'both correct'}
        encoded_results = torch.sum(sigma_diffs * torch.tensor([1., 2.]).to(self.device), dim=1)
        sigma_acc = (encoded_results == 3).sum() / targets.size(0)
        new_row = pd.DataFrame([{
            'Scope': 'Entire Model',
            'Class': 'All',
            'Accuracy': cls_acc * 100,
            'Sigma Accuracy': sigma_acc.item() * 100,
            'Amount': targets.size(0),
        }])

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        print(f"Total Class accuracy: {cls_acc * 100:.2f}%")
        print(f"Total Sigma accuracy: {sigma_acc * 100:.2f}%")
        print('-----------------------------------------------')

        print("Each Class Analysis: ")
        for cls in self.classes_indices:
            cls_idx = torch.where(cls_targets == cls)[0]
            cls_predictions = predictions[cls_idx]
            cls_sigmas = sigmas[cls_idx]
            cls_sigma_targets = sigma_targets[cls_idx]
            cls_acc = (cls_predictions == cls).sum().item() / cls_idx.size(0)
            sigma_diffs = (cls_sigmas == cls_sigma_targets)
            encoded_results = torch.sum(sigma_diffs * torch.tensor([1., 2.]).to(self.device), dim=1)
            sigma_acc = (encoded_results == 3).sum().item() / cls_idx.size(0)
            new_row = pd.DataFrame([{
                'Scope': 'Entire Model',
                'Class': CLASSES_NAMES[self.dataset_name][cls],
                'Accuracy': cls_acc * 100,
                'Sigma Accuracy': sigma_acc * 100,
                'Amount': cls_idx.size(0),
            }])

            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

            print(f"Class: {CLASSES_NAMES[self.dataset_name][cls]}")
            print(f"Accuracy: {cls_acc * 100:.2f}%", end=', ')
            print(f"Sigma accuracy: {sigma_acc * 100:.2f}%")

        branch_num = 2 * sigmas.size(1)
        sigma_encoding = {0: 'branch_0', 1: 'branch_2', 2: 'branch_1', 3: 'branch_3'} if branch_num == 4 else {0: 'branch_0', 3: 'branch_1'}
        encoded_sigma_results = torch.sum(sigmas * torch.tensor([1., 2.]).to(self.device), dim=1)
        if sigmas.size(1) == 1:
            pass
            # self.plot_class_analysis(predictions, cls_targets, sigmas, sigma_targets)
        # Each Branch Analysis
        print("Each Branch Analysis: ")
        for ii in range(branch_num):
            tt = ii
            if branch_num == 2:
                tt = ii * 3
            print('-----------------------------------------------')
            branch_idx = torch.where(encoded_sigma_results == tt)[0]
            branch_predictions = predictions[branch_idx]
            branch_cls_targets = cls_targets[branch_idx]
            branch_sigmas = sigmas[branch_idx]
            branch_sigmas_targets = sigma_targets[branch_idx]
            branch_cls_acc = (branch_predictions == branch_cls_targets).sum().item() / branch_cls_targets.size(0) if branch_cls_targets.size(0) > 0 else 0.0
            branch_sigma_diffs = (branch_sigmas == branch_sigmas_targets)

            encoded_results = torch.sum(branch_sigma_diffs * torch.tensor([1., 2.]).to(self.device), dim=1)
            branch_sigma_acc = (encoded_results == 3).sum().item() / branch_cls_targets.size(0) if branch_cls_targets.size(0) > 0 else 0.0
            new_row = pd.DataFrame([{
                'Scope': f'Branch_{ii}',
                'Class': 'All',
                'Accuracy': branch_cls_acc * 100,
                'Sigma Accuracy': branch_sigma_acc * 100,
                'Amount': branch_cls_targets.size(0),
            }])

            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

            print(f"Branch_{ii} Class accuracy: {branch_cls_acc * 100:.2f}%")
            print(f"Branch_{ii} Sigma accuracy: {branch_sigma_acc * 100:.2f}%")

            for cls in self.classes_indices:
                class_name = CLASSES_NAMES[self.dataset_name][cls]
                cls_idx = torch.where(branch_cls_targets == cls)[0]
                cls_acc = (branch_predictions[cls_idx] == cls).sum().item() / cls_idx.size(0) if cls_idx.size(0) > 0 else 0
                sigma_diffs = (branch_sigmas[cls_idx] == branch_sigmas_targets[cls_idx])
                encoded_results = torch.sum(sigma_diffs * torch.tensor([1., 2.]).to(self.device), dim=1)
                branch_sigma_acc = (encoded_results == 3).sum().item() / cls_idx.size(0) if cls_idx.size(0) > 0 else 0.0
                new_row = pd.DataFrame([{
                    'Scope': f'Branch_{ii}',
                    'Class': class_name,
                    'Accuracy': cls_acc * 100,
                    'Sigma Accuracy': branch_sigma_acc * 100,
                    'Amount': cls_idx.size(0),
                }])

                metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

                print(f"Class: {class_name}, Amount: {cls_idx.size(0)}")
                print(f"Accuracy: {cls_acc * 100:.2f}%", end=', ')
                print(f"Sigma accuracy: {branch_sigma_acc * 100:.2f}%")

        print('-----------------------------------------------')
        encoded_sigma_targets = torch.sum(sigma_targets * torch.tensor([1., 2.]).to(self.device), dim=1)
        if branch_num == 2:
            sigmas_r_list = [sigmas_r[encoded_sigma_targets == s][:, 0].cpu().numpy() for s, code in sigma_encoding.items()]
        else:
            sigmas_r_list = [(sigmas_r[:, 0]*sigmas_r[:, 1])[encoded_sigma_targets == s].cpu().numpy() for s, code in
                             sigma_encoding.items()]
        labels = [
            "Embedding For The First Branch\Group",
            "Embedding For The Second Branch\Group",
            "Embedding For The Third Branch\Group",
            "Embedding For The Fourth Branch\Group"
        ]
        plot_title = 'Density Plot of Scalars For The Branches'
        self.plot_density(plot_title, labels, *sigmas_r_list)
        if branch_num == 2:
            sigmas_b_r_list = [sigmas_b_r[encoded_sigma_targets == s][:, 0].cpu().numpy() for s, code in sigma_encoding.items()]
        else:
            sigmas_b_r = sigmas_b_r.to(self.device)  # Ensure sigmas_b_r is on the correct device
            encoded_sigma_targets = encoded_sigma_targets.cpu()  # Ensure consistency
            # s = s.to(self.device)
            sigmas_b_r_list = [
                self.branches_probabilities(encoded_sigma_targets, sigmas_b_r)[encoded_sigma_targets == s][:,
                0].cpu().numpy() for s, code in sigma_encoding.items()]

        labels = [
            "Routing Probability For The First Branch",
            "Routing Probability For The Second Branch",
            "Routing Probability For The Third Branch",
            "Routing Probability For The Fourth Branch"
        ]
        plot_title = 'Density Plot of Routing Probability For The Branches'
        # wrong_routing_sigmas_b_r = [sigmas_b_r[(encoded_sigma_targets == s) & (sigmas != sigma_targets).squeeze(1)][:, 0].cpu().numpy() for
        #         s, code in sigma_encoding.items()]
        self.plot_density(plot_title, labels, *sigmas_b_r_list)
        return metrics_df

    def plot_density(self, plot_title, labels, *args):
        colors = ["blue", "red", "green", "purple"]

        plt.figure(figsize=(10, 6))

        for i, arg in enumerate(args):
            sns.histplot(arg, color=colors[i], label=labels[i], kde=False)

        plt.xlabel('Scalar Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(plot_title, fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()
    def plot_class_analysis(self, predictions, cls_targets, sigmas, sigma_targets):
        num_images = 0
        for cls in self.classes_indices:
            if num_images % 10 == 0:
                plt.figure(figsize=(15, 8))

            plt.subplot(5, 2, num_images % 10 + 1)

            class_name = CLASSES_NAMES[self.dataset_name][cls]
            plt.title(class_name)

            cls_idx = torch.where(cls_targets == cls)[0]
            cls_acc = (predictions[cls_idx] == cls).sum().item() / cls_idx.size(0)
            sigma_diffs = (sigmas[cls_idx] == sigma_targets[cls_idx])
            encoded_results = torch.sum(sigma_diffs * torch.tensor([1., 2.]).to(self.device), dim=1)
            sigma_acc = (encoded_results == 3).sum() / sigmas.size(0)

            encoding = {0: 'wrong routing', 1: 'correct routing'}
            encoded_results = sigma_diffs.sum(dim=1).to(self.device)
            results = {label: (encoded_results == code).sum().item() for code, label in encoding.items()}
            plt.bar(results.keys(), results.values())

            num_images += 1

        plt.tight_layout()
        plt.show()

    def plot_branch_analysis(self, predictions, cls_targets, sigmas, sigmas_r):
        branch_num = 2
        for ii in range(branch_num):
            branch_predictions = predictions[sigmas == ii]
            branch_cls_targets = cls_targets[sigmas == ii]
            branch_cls_acc = (branch_predictions == branch_cls_targets).sum().item() / branch_cls_targets.size(0)
            print(f"Branch_{ii} Class accuracy: {branch_cls_acc * 100:.2f}%")

            num_images = 0
            for cls in self.classes_indices:
                class_name = CLASSES_NAMES[self.dataset_name][cls]
                plt.title(class_name)

                cls_idx = torch.where(branch_cls_targets == cls)[0]
                cls_acc = (branch_predictions[cls_idx] == cls).sum().item() / cls_idx.size(0)
                print(f"Accuracy: {cls_acc * 100:.2f}%")

        self.plot_density(sigmas_r[sigmas == 0].cpu().numpy(), sigmas_r[sigmas == 1].cpu().numpy())

    def branches_probabilities(self, encoded_sigma_results, data_tensor):

        output_tensor = torch.full((data_tensor.size(0), 1), -1.0)

        for i in range(encoded_sigma_results.size(0)):
            val = encoded_sigma_results[i].item()

            if val == 0:
                output_tensor[i] = (1 - data_tensor[i, 0]) * (1 - data_tensor[i, 1])
            elif val == 1:
                output_tensor[i] = data_tensor[i, 0] * (1 - data_tensor[i, 1])
            elif val == 2:
                output_tensor[i] = (1 - data_tensor[i, 0]) * data_tensor[i, 1]
            elif val == 3:
                output_tensor[i] = data_tensor[i, 0] * data_tensor[i, 1]

        return output_tensor
    # noinspection PyTypeChecker
    def sigma_analysis(self):
        activations_dict = {}
        self.register_hooks(activations_dict)
        super().evaluate()
        targets = torch.tensor(self.datasets.test_set.targets).to(self.device)
        sigmas = activations_dict['sigma'].round()
        cls_targets = targets[:, 0]
        sigma_targets = targets[:, 1:]

        removed = []
        new_labels_map = {k: v[1:] for k, v in LABELS_MAP[self.dataset_name].items()}
        for s in range(2):
            soft_clustering_cands = []
            for cls in self.classes_indices:
                if cls in removed:
                    continue
                class_name = CLASSES_NAMES[self.dataset_name][cls]
                cls_idx = torch.where(cls_targets == cls)[0]
                results = []
                for i in range(2):
                    correct_until_split_idx = \
                        torch.where(torch.all(sigma_targets[cls_idx, :s] == sigmas[cls_idx, :s], dim=1))[0]
                    num_i = torch.sum(sigmas[cls_idx[correct_until_split_idx], s] == i).item()
                    results.append(num_i)
                if min(results) > 0.2 * sum(results):
                    soft_clustering_cands.append((class_name, cls, results))
                    plt.figure()
                    plt.bar(['0', '1'], results)
                    plt.title(f'{class_name} - split {s}')
            plt.show()
            print(f"SOFT CLUSTERING CANDS - SPLIT {s}:")
            for cls_name, cls, res in soft_clustering_cands:
                print(cls_name, f'({cls})', '- results:', res)
                orig_labels = new_labels_map[cls]
                new_labels = orig_labels[:s] + (0.5,) * (len(orig_labels) - s)
                new_labels_map[cls] = new_labels
                removed.append(cls)
        print(new_labels_map)

    def _init_sigma_weights(self):
        num_sigma_labels_per_samples = len(self.datasets.train_set.targets[0]) - 1
        sigma_weights = torch.tensor([0.5 ** i for i in range(num_sigma_labels_per_samples)]).to(self.device)
        return sigma_weights


class NIN_HVT_Trainer(HVT_Trainer):

    def _init_model(self):
        set_random_seed(0)
        model = NIN_HyperDecisioNet(input_channels=INPUT_SIZE[self.dataset_name][0])
        # model.apply(functools.partial(weights_init_kaiming, scale=0.01))
        # model.apply(self.weights_init_xavier)
        return model

    def init_parser(self):
        parser = super().init_parser()
        return parser

    def _init_config_attributes(self):
        super()._init_config_attributes()

    def init_data_sets(self):
        labels_map = dict(LABELS_MAP[self.dataset_name])
        num_blocks = 2
        for k, v in labels_map.items():
            labels_map[k] = v[:num_blocks]
        return FilteredRelabeledDatasets(self.transforms, use_validation=self.use_validation,
                                         classes_indices=self.classes_indices,
                                         labels_map=labels_map,
                                         dataset_name=self.dataset_name)

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

class WideResNet_HVT_Trainer(HVT_Trainer):

    def init_transforms(self, padding_mode='constant'):
        return super().init_transforms(padding_mode='reflect')

    def init_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [60, 120, 160], gamma=0.2, verbose=True)
    def lr_scheduler_step(self, epoch=-1, train_acc=None, train_loss=None, test_acc=None, test_loss=None):
        self.lr_scheduler.step()

    def init_optimizer(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                    momentum=0.9, weight_decay=5e-4, nesterov=True)
        return optimizer

    def _init_model(self):
        num_in_channels = INPUT_SIZE[self.dataset_name][0]
        num_classes = NUM_CLASSES[self.dataset_name]
        if self.wrn_cfg_name == '100_baseline':
            stage_sizes = WRESNET_STAGE_SIZES['100_baseline'][0]
            model = WideResNet_HyperDecisioNet_2_split(28, 10, dropout_p=0.3, num_classes=num_classes,
                                                       num_in_channels=num_in_channels, stage_sizes=stage_sizes)
        else:
            stage_sizes = WRESNET_STAGE_SIZES['100_baseline_single_early'][0]
            model = WideResNet_HyperDecisioNet_1_split(28, 10, dropout_p=0.3, num_classes=num_classes,
                                                       num_in_channels=num_in_channels, stage_sizes=stage_sizes)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return model

    def init_parser(self):
        parser = super().init_parser()
        parser.add_argument('--wrn_cfg_name', type=str, default='100_baseline', help='Name of the Wide-ResNet config')
        return parser

    def _init_config_attributes(self):
        super()._init_config_attributes()
        self.wrn_cfg_name = self.config['wrn_cfg_name']

    def init_data_sets(self):
        if self.dataset_name == "FashionMNIST":
            labels_map = dict(LABELS_MAP[f'{self.dataset_name}'])
        else:
            labels_map = dict(LABELS_MAP[f'{self.dataset_name}_WRN'])
        num_levels_in_tree = WRESNET_STAGE_SIZES[self.wrn_cfg_name][1]
        for k, v in labels_map.items():
            labels_map[k] = v[:num_levels_in_tree]
        return FilteredRelabeledDatasets(self.transforms, use_validation=self.use_validation,
                                         classes_indices=self.classes_indices,
                                         labels_map=labels_map,
                                         dataset_name=self.dataset_name)

    def _feed_forward(self, inputs, targets):
        cls_targets, *sigma_targets = targets
        sigma_targets = torch.column_stack(sigma_targets)
        binarize = self.always_binarize or random.random() > 0.5
        outputs, sigma_b, sigma_r, _ = self.model(inputs, binarize=binarize)
        sigma_b = sigma_b.unsqueeze(1) if sigma_b.dim() == 1 else sigma_b
        # print(f'Cls correct: {sum(torch.argmax(outputs, 1)==cls_targets)}')
        # print(f'Sigma correct: {sum(sigmas==sigma_targets)}\n')
        cls_loss = self.cls_criterion(outputs, cls_targets.long())
        sigma_loss = self.sigma_criterion(sigma_b, sigma_targets.float())
        combined_loss = cls_loss + self.beta * sigma_loss
        self.metrics_tracker.update(cls_loss, sigma_loss, combined_loss, outputs, cls_targets, sigma_b, sigma_targets)
        return outputs, combined_loss

if __name__ == '__main__':
    trainer = WideResNet_HVT_Trainer()
    # trainer = NIN_HVT_Trainer()
    # trainer = wide_resnet
    # trainer.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(trainer.device)
    # import torch

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    # trainer = NIN_HVT_Trainer()
    input_tensor = torch.randn(1, 3, 32, 32).to(trainer.device)
    params_num = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f'Number Of Parameters: {params_num}')
    trainer.train_model()
    results = trainer.evaluate()
    # experiment_name = f"{trainer.dataset_name}_{trainer.config['exp_name']}_params_num_{params_num}"
    # prepare_output_to_local(results, experiment_name)
