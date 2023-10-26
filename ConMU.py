import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from torch.utils.data import ConcatDataset, DataLoader

from dataset import CustomImageDataset
from utils import pruning_model_random, pruning_model, remove_prune, check_sparsity

sys.path.append(('../'))
sys.path.append(('../../'))
import evaluation_metrics
import utils
import time


def further_train(model, incompetent_model, test_loader, retain_loader, forget_loader, device, unlearning_time, args):
    # further_train time start
    further_train_start_time = time.time()

    if args.random_prune:
        print("random pruning")
        pruning_model_random(model, args.rate)
    else:
        print("L1 pruning")
        pruning_model(model, args.rate)
    remove_prune(model)

    check_sparsity(model)

    if (isinstance(model, torch.nn.Module) and "ResNet" in model.__class__.__name__) or (
            isinstance(model, torch.nn.Module) and "vgg" in model.__class__.__name__.lower()):

        print("len of forget_loader: ", len(forget_loader))
        print("len of retain_loader: ", len(retain_loader))

        important_forget_data, _, _, forget_data_time = select_important_data(forget_loader, model, args, device,
                                                                              retain_loader=False)
        important_retain_data, _, _, retain_data_time = select_important_data(retain_loader, model, args, device,
                                                                              retain_loader=True)

        print("len of important_forget_data: ", len(important_forget_data), "with time: ", forget_data_time)
        print("len of important_retain_data: ", len(important_retain_data), "with time: ", retain_data_time)

        noised_forget_loader = add_noise(important_forget_data, args.num_noise, args)

        print("len of noised_forget_loader: ", len(noised_forget_loader))

        # combine important_retain_loader and noised_forget_loader
        combined_loader = DataLoader(ConcatDataset([important_retain_data.dataset, noised_forget_loader.dataset]),
                                     batch_size=args.batch_size, shuffle=True)

        print("len of combined_loader: ", len(combined_loader))

        # further train the model on combined_loader
        model = model.to(device)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, args.further_train_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        incompetent_model = incompetent_model.to(device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        further_train_start_time = time.time()
        model.train()
        incompetent_model.eval()
        temperature = args.temperature  # Define the temperature value
        kl_weight = args.kl_weight  # This is the weight for the KL loss

        for epoch in range(args.further_train_epoch):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(combined_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                scaled_outputs = outputs / temperature
                scaled_outputs_incompetent = incompetent_model(inputs) / temperature
                outputs_incompetent = F.log_softmax(scaled_outputs_incompetent, dim=1)
                KL_Loss = F.kl_div(outputs_incompetent, F.softmax(scaled_outputs, dim=1), reduction='batchmean')

                loss += kl_weight * KL_Loss
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            scheduler.step()
            test_acc = utils.evaluate_acc(model, test_loader, device)
            print(
                f"Further Training Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Train, Accuracy = {accuracy:.2f}%, Test Accuracy = {test_acc:.2f}%")
        further_train_end_time = time.time()
        further_train_time = further_train_end_time - further_train_start_time
        print("furhter train time: ", further_train_time)
        evaluation_result = evaluation_metrics.MIA_Accuracy(model=model,
                                                            forget_loader=forget_loader,
                                                            retain_loader=retain_loader,
                                                            test_loader=test_loader,
                                                            device=device,
                                                            total_unlearn_time=unlearning_time + further_train_time,
                                                            args=args)
        return evaluation_result
    else:
        # raise exceptions saying other models are not supported yet
        raise NotImplementedError("Only ResNet and VGG are supported for now")


# create a method called add_noise that takes in data_loader, and adds Gaussian noise to the data_loader
def add_noise(data_loader, noise_level, args):
    noisy_data = []
    noisy_label = []
    for i, (X, y) in enumerate(data_loader):
        X = X + noise_level * torch.randn_like(X)
        noisy_data.extend([x for x in X])  # Flatten the data
        noisy_label.extend([label for label in y])  # Flatten the labels
    noisy_loader = DataLoader(CustomImageDataset(noisy_data, noisy_label), batch_size=args.batch_size,
                              shuffle=True)
    return noisy_loader


def select_important_data(data_loader, model, args, device, retain_loader=False):
    utils.setup_seed(42)
    start_time = time.time()
    model.eval()
    model = nn.DataParallel(model)
    model = model.to(device)

    # Store L2 normed loss for each individual data sample
    l2_losses = []

    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_pred = F.softmax(y_hat.float(), dim=1)
            num_classes = y_pred.shape[1]
            y_one_hot = F.one_hot(y.to(torch.int64), num_classes).float()

            l2_loss = y_pred - y_one_hot
            norm_loss = LA.norm(l2_loss, ord=2, dim=1)
            l2_losses.extend(norm_loss.tolist())  # Store individual L2 normed losses

    # Compute global statistics
    mean = np.mean(l2_losses)
    std = np.std(l2_losses)

    # Determine the bounds
    if retain_loader:
        upper_bound = mean + args.retain_filter_up * std
        lower_bound = mean - args.retain_filter_lower * std
    else:
        upper_bound = mean + args.forget_filter_up * std
        lower_bound = mean - args.forget_filter_lower * std

    # Select important samples
    important_x = []
    important_y = []
    for i, (X, y) in enumerate(data_loader.dataset):
        if l2_losses[i] > lower_bound and l2_losses[i] < upper_bound:
            important_x.append(X)
            important_y.append(y)

    important_loader = DataLoader(CustomImageDataset(important_x, important_y), batch_size=args.batch_size,
                                  shuffle=True)
    end_time = time.time()
    return important_loader, mean, std, end_time - start_time
