import sys

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import utils

sys.path.append(('../'))
sys.path.append(('../../'))
from utils import *
import time


def data_init(args):
    utils.setup_seed(42)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == 'svhn':

        model, incompetent_model, train_loader, _, test_loader, marked_loader = setup_model_dataset(args)

        forget_loader, retain_loader = utils.split_train_to_forget_retain(marked_loader=marked_loader,
                                                                          forget_percentage=args.forget_percentage,
                                                                          batch_size=args.batch_size)

    else:
        raise Exception("Other datasets are not supported yet")

    incompetent_model = incompetent_model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(incompetent_model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # record the time for training
    start_time = time.time()
    # train incomepetent model using retain_loader for three epochs with the scheduler
    for epoch in range(args.incompetent_epoch):
        loss, acc = utils.train_one_epoch(incompetent_model, retain_loader, criterion, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {acc:.2f}%")
    end_time = time.time()
    total_unlearn_time = end_time - start_time

    return model, incompetent_model, train_loader, test_loader, forget_loader, retain_loader, total_unlearn_time
