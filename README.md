# Controallble Machine Unlearning

## Introduction
Here is the Controllable Machine Unlearning (ConMU) repository. We conduct our experiments on CV benchmarks
for both the random forgetting and class-wise forgetting.

## Random Forgetting

Here is the command to perform ConMU's random forgetting using ResNet-18 on CIFAR10 dataset.

```angular2html
python -u main.py --batch_size 128 --data CV_Data/cifar10 --dataset cifar10 --num_classes 10 --arch resnet18 --forget_percentage 0.2 
--save_dir ./dataset/resnet18 --epochs 100 --retrain_epoch 80 --lr 1e-3 --retrain_lr 1e-3 --prune_type rewind_lt --retain_filter_up 0.314 
--retain_filter_lower 0.314 --forget_filter_up 0.312 --forget_filter_lower 0.314 --num_noise 5 --further_train_lr 1e-2 --further_train_epoch 5 
--incompetent_epoch 3 --kl_weight 1.0
```

### Parameters
- batch_size: batch size for the data
- data: location of the data corpus
- dataset: name of the dataset (cifar10, cifar100, svhn)
- num_classes: number of classes in the dataset
- arch: model architecture (Refer to [cv_models/__init__.py](cv_models/__init__.py)).
- forget_percentage: percentage of data to forget
- save_dir: directory to save the model checkpoints
- epochs: number of epochs to train the original model
- retrain_epoch: number of epochs to train the gold model (retrained model using retained dataset)
- lr: learning rate for the original model
- retrain_lr: learning rate for the gold model
- num_noise: amount of noise (standard Gaussian) to add to the forgotten datasets
- further_train_lr: learning rate for the ConMU
- further_train_epoch: number of epochs to train the ConMU
- incompetent_epoch: number of epochs to train the unlearning proxy model

More details be be found at [arg_parser.py](arg_parser.py).


## Class-wise Forgetting
Here is the command to perform ConMU's class-wise forgetting using ResNet-18 on CIFAR10 dataset.
```angular2html
python -u main.py --batch_size 128 --data CV_Data/cifar10 --dataset cifar10 --num_classes 10 --arch resnet18 --forget_percentage 0.5 
--save_dir ./dataset/resnet18 --epochs 100 --retrain_epoch 80 --lr 1e-3 --retrain_lr 1e-3 --prune_type rewind_lt --retain_filter_up 0.314 
--retain_filter_lower 0.5 --forget_filter_up 0.5 --forget_filter_lower 0.5 --num_noise 5 --further_train_lr 1e-2 --further_train_epoch 5 
--incompetent_epoch 3 --kl_weight 0.5 --class_wise 5
```

Note that the the main differences between random forgetting and class-wise forgetting is that the ```--class_wise``` parameter is added to the command, which
specifies the class number that we want to forget. Also, the default ```--forget_percentage``` is 0.5, which means that we forget 50% of the data from the specified class.


