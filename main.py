import copy
import sys

import torch.nn as nn
import torch.optim
import torch.utils.data

import utils
from data_initialization import data_init

sys.path.append(('../'))
sys.path.append(('../../'))
import torch.optim
import torch.utils.data
import sys
import time
import evaluation_metrics

sys.path.append(('../'))
sys.path.append(('../../'))
import torch
import arg_parser
import ConMU

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def model_training(model_copy, train_loader, test_loader, args, original=True):
    model_copy = model_copy.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    if original:
        epochs = args.epochs
        lr = args.lr
    else:
        epochs = args.retrain_epoch
        lr = args.retrain_lr
    optimizer = torch.optim.SGD(model_copy.parameters(), lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    utils.train_model(model=model_copy,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      criterion=criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      epochs=epochs,
                      device=device)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = arg_parser.parse_args()

    model, incompetent_model, train_loader, test_loader, forget_loader, retain_loader, total_unlearn_time = data_init(
        args)

    # ======================================== Original Model ========================================
    start_time = time.time()
    model_copy = copy.deepcopy(model)
    model_training(model_copy, train_loader, test_loader, args, original=True)
    end_time = time.time()
    original_model_training_time = end_time - start_time
    print("model training (original_model) time: ", original_model_training_time)
    original_evaluate_result = evaluation_metrics.MIA_Accuracy(model=model_copy,
                                                               forget_loader=forget_loader,
                                                               retain_loader=retain_loader,
                                                               test_loader=test_loader,
                                                               device=device,
                                                               total_unlearn_time=original_model_training_time,
                                                               args=args)
    utils.save_checkpoint({
        'state_dict': model_copy.state_dict(),
    }, save_path=args.save_dir, filename='_original_model_checkpoint.pth.tar')
    print("original_evaluate_result: ", original_evaluate_result)

    # ======================================== ConMU ========================================

    model_further_train = copy.deepcopy(model)
    model_checkpoint = utils.load_checkpoint(device, args.save_dir, filename='_original_model_checkpoint.pth.tar')

    model_further_train.load_state_dict(model_checkpoint["state_dict"])
    evaluation_result = ConMU.further_train(model=model_further_train,
                                            incompetent_model=incompetent_model,
                                            test_loader=test_loader,
                                            retain_loader=retain_loader,
                                            forget_loader=forget_loader,
                                            device=device,
                                            unlearning_time=total_unlearn_time,
                                            args=args)
    print("ConMU further train result: ", evaluation_result)

    # ======================================== Retrain ========================================
    start_time = time.time()
    model_copy_retrain = copy.deepcopy(model)
    model_training(model_copy_retrain, retain_loader, test_loader, args, original=False)
    end_time = time.time()
    retrain_model_training_time = end_time - start_time
    print("model training (retraining) time: ", retrain_model_training_time)
    retrain_evaluate_result = evaluation_metrics.MIA_Accuracy(model=model_copy_retrain,
                                                              forget_loader=forget_loader,
                                                              retain_loader=retain_loader,
                                                              test_loader=test_loader,
                                                              device=device,
                                                              total_unlearn_time=retrain_model_training_time,
                                                              args=args)
    utils.save_checkpoint({
        'state_dict': model_copy_retrain.state_dict(),
    }, save_path=args.save_dir, filename='_retrain_model_checkpoint.pth.tar')
    print("retrain_evaluate_result: ", retrain_evaluate_result)


if __name__ == '__main__':
    main()
