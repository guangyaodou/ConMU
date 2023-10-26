import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Lottery Tickets Experiments')

    ##################################### Dataset #################################################
    # parser.add_argument('--data', type=str, default='../data',
    #                     help='location of the data corpus')
    parser.add_argument('--data', type=str, default='Tab_Data/Adult/adult.csv',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str,
                        default='cifar10', help='[Adult, Breast_Cancer, Diabetes]')
    parser.add_argument('--data_dir', type=str,
                        default='./tiny-imagenet-200', help='dir to tiny-imagenet')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    ##################################### Architecture ############################################
    parser.add_argument('--arch', type=str,
                        default='resnet18', help='model architecture')
    parser.add_argument('--imagenet_arch', action="store_true",
                        help="architecture for imagenet size samples")

    ##################################### General setting ############################################
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--train_seed', default=1, type=int,
                        help='seed for training (default value same as args.seed)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--workers', type=int, default=3,
                        help='number of workers in dataloader')
    parser.add_argument('--checkpoint', type=str,
                        default=None, help='checkpoint file')
    parser.add_argument(
        '--save_dir', help='The directory used to save the trained models', default=None, type=str)

    ##################################### Training setting #################################################
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='weight decay')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run for the original model')
    parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')

    ##################################### Pruning setting #################################################
    parser.add_argument('--prune', type=str, default="omp",
                        help="method to prune")
    parser.add_argument('--rate', default=0.1, type=float,
                        help='pruning rate')  # pruning rate is always 20%
    parser.add_argument('--prune_type', default='rewind_lt', type=str,
                        help='IMP type (lt, pt or rewind_lt)')
    parser.add_argument('--random_prune', action='store_true',
                        help='whether using random prune')

    ##################################### Unlearn setting #################################################
    parser.add_argument('--unlearn', type=str,
                        default='retrain', help='method to unlearn')
    parser.add_argument('--incompetent_epoch', type=int,
                        default=1, help='epoch to train the unleanring proxy model')
    parser.add_argument('--class_wise', type=str,
                        default=None, help='class to forget')

    parser.add_argument('--retain_filter_up', type=float,
                        default=0.5, help='retain_filter_up * std is upper bound for retain data filter')
    parser.add_argument('--retain_filter_lower', type=float,
                        default=0.5, help='retain_filter_lower * std is lower bound for retain data filter')

    parser.add_argument('--forget_filter_up', type=float,
                        default=0.5, help='forget_filter_up * std is the upper bound for forget data filter')
    parser.add_argument('--forget_filter_lower', type=float,
                        default=0.5, help='forget_filter_lower * std is the lower bound for forget data filter')

    # parser.add_argument('--class_forget', type=str,
    #                     default='N/A', help='choose majority (maj) or minority (min) class to forget under class_wise forgeting setup')
    parser.add_argument('--forget_percentage', type=float,
                        default=0.2, help='forget percentage for forget loader')
    parser.add_argument('--further_train_epoch', type=int,
                        default=10, help='rewinding epoch during further train')
    parser.add_argument('--further_train_lr', type=float,
                        default=1e-4, help='learning rate during further train')
    parser.add_argument('--unlearn_lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--unlearn_epochs', default=150, type=int,
                        help='number of total epochs for unlearn to run')
    parser.add_argument('--num_indexes_to_replace', type=int, default=None,
                        help='Number of data to forget')
    parser.add_argument('--class_to_replace', type=int, default=0,
                        help='Specific class to forget')
    parser.add_argument('--indexes_to_replace', type=list, default=None,
                        help='Specific index data to forget')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='unlearn noise')
    parser.add_argument('--num_noise', default=2, type=int,
                        help='num of noise added to image')
    parser.add_argument('--kl_weight', type=float, default=0.5,
                        help='the weight of KL Loss during further training')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help="temperature for KL loss")

    ##################################### retrain setting #################################################
    parser.add_argument('--retrain_epoch', type=int,
                        default=10, help='number of epochs for retrian using only the retained data')
    parser.add_argument('--retrain_lr', default=1e-4, type=float,
                        help='retrain learning rate')
    return parser.parse_args()
