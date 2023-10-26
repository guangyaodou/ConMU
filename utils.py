import os
import random
import sys

import numpy as np
import torch
import torch.nn.utils.prune as prune
from torch import nn
from torchvision import transforms

import arg_parser
from cv_models import *
from dataset import *

__all__ = ['setup_model_dataset', 'AverageMeter',
           'save_checkpoint', 'setup_seed', 'accuracy']
sys.path.append(('../'))
sys.path.append(('../../'))
from torch.utils.data import Subset


def check_sparsity(model):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    if zero_sum:
        remain_weight_ratie = 100 * (1 - zero_sum / sum_list)
        print('* remain weight ratio = ', 100 * (1 - zero_sum / sum_list), '%')
    else:
        print('no weight for calculating sparsity')
        remain_weight_ratie = None

    return remain_weight_ratie


def pruning_model_random(model, px):
    print('Apply Unstructured Random Pruning Globally (all conv layers)')
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            parameters_to_prune.append((m, 'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )


def remove_prune(model):
    print('Remove hooks for multiplying masks (all conv layers and Linear layers)')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.remove(m, 'weight')


# Pruning operation
def pruning_model(model, px):
    print('Apply Unstructured L1 Pruning Globally (all conv layers and all linear layers)')
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m, 'weight'))
        elif isinstance(m, nn.Linear):
            parameters_to_prune.append((m, 'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)


def load_checkpoint(device, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


def setup_seed(seed):
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_model_dataset(args):
    if args.dataset == 'cifar10':
        setup_seed(args.train_seed)
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_full_loader, val_loader, test_loader = cifar10_dataloaders_no_val(batch_size=args.batch_size,
                                                                                data_dir=args.data,
                                                                                num_workers=args.workers)
        marked_loader, _, _ = cifar10_dataloaders_no_val(batch_size=args.batch_size,
                                                         data_dir=args.data,
                                                         num_workers=args.workers)
        if args.train_seed is None:
            args.train_seed = args.seed

        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
            incompetent_model = copy.deepcopy(model)

        else:
            model = model_dict[args.arch](num_classes=classes)
            incompetent_model = copy.deepcopy(model)

        model.normalize = normalization
        incompetent_model.normalize = normalization
        print(model)
        return model, incompetent_model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == 'svhn':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        train_full_loader, val_loader, _ = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        marked_loader, _, test_loader = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
            incompetent_model = copy.deepcopy(model)

        else:
            model = model_dict[args.arch](num_classes=classes)
            incompetent_model = copy.deepcopy(model)

        model.normalize = normalization
        print(model)
        return model, incompetent_model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == 'cifar100':
        setup_seed(args.train_seed)
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_full_loader, val_loader, test_loader = cifar100_dataloaders_no_val(batch_size=args.batch_size,
                                                                                 data_dir=args.data,
                                                                                 num_workers=args.workers)
        marked_loader, _, _ = cifar100_dataloaders_no_val(batch_size=args.batch_size,
                                                          data_dir=args.data,
                                                          num_workers=args.workers)
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
            incompetent_model = copy.deepcopy(model)
        else:
            model = model_dict[args.arch](num_classes=classes)
            incompetent_model = copy.deepcopy(model)

        model.normalize = normalization
        print(model)
        return model, incompetent_model, train_full_loader, val_loader, test_loader, marked_loader
    else:
        raise ValueError('Dataset not supprot yet !')


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        test_transform = transforms.Compose([
        ])
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def split_train_to_forget_retain(marked_loader, forget_percentage, batch_size, seed_=42):
    # Set the random seed for reproducibility
    torch.manual_seed(seed_)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_)

    args = arg_parser.parse_args()

    whole_dataset = marked_loader.dataset
    if args.dataset == 'svhn':
        label = whole_dataset.labels
    else:
        label = whole_dataset.targets

    if args.class_wise != None:
        if (args.class_wise).isdigit() and int(args.class_wise) in list(label):
            forget_indices = [idx for idx, label in enumerate(label) if label == int(args.class_wise)]
            forget_class_indices = [i for i in
                                    random.sample(forget_indices, int(len(forget_indices) * forget_percentage))]
            forget_dataset = Subset(whole_dataset, indices=forget_class_indices)
            retain_dataset = Subset(whole_dataset, indices=[i for i in range(len(whole_dataset)) if
                                                            i not in forget_class_indices])

        else:
            raise Exception("This class is not in the designated dataset!")
    else:
        split_point = int(len(whole_dataset) * forget_percentage)
        forget_dataset = Subset(whole_dataset, indices=[i for i in range(split_point)])
        retain_dataset = Subset(whole_dataset, indices=[i for i in range(split_point, len(whole_dataset))])

    forget_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = torch.utils.data.DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    assert len(forget_dataset) + len(retain_dataset) == len(marked_loader.dataset)
    print("Completed splitting dataset into forget and retain")

    return forget_loader, retain_loader


# define evaluate_acc that returns the accuracy of the model on the data set
def evaluate_acc(model, data_loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / (batch_idx + 1)
    accuracy = 100. * correct / total

    # Step the scheduler after each epoch
    scheduler.step()

    return avg_loss, accuracy


# train a model for epochs times using train_one_epoch, and report avg loss, train accuracy, and test accuracy for every 5 epochs,
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device):
    for epoch in range(epochs):
        # Train for one epoch
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)

        # Evaluate the model on the test set every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_acc = evaluate_acc(model, test_loader, device)
            print(
                f"Epoch {epoch + 1}: Loss = {loss:.4f}, Train Accuracy = {train_acc:.2f}%, Test Accuracy = {test_acc:.2f}%")
