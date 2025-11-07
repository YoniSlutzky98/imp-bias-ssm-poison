"""Implement an ArgParser common to both brew_poison.py and dist_brew_poison.py ."""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')


    ###########################################################################
    # Central:
    parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
    parser.add_argument('--recipe', default='gradient-matching', type=str, choices=['gradient-matching', 'gradient-matching-private',
                                                                                    'watermarking', 'poison-frogs', 'metapoison', 'bullseye'])
    parser.add_argument('--threatmodel', default='random-subset-random-targets', type=str, choices=['single-class', 'third-party', 'random-subset', 'random-subset-random-targets'])

    # Reproducibility management:
    parser.add_argument('--poisonkey', default=None, type=str, help='Initialize poison setup with this key.')  # Also takes a triplet 0-3-1
    parser.add_argument('--modelkey', default=None, type=int, help='Initialize the model with this key.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')

    # Poison properties / controlling the strength of the attack:
    parser.add_argument('--eps', default=16, type=float)
    parser.add_argument('--budget', default=0.2, type=float, help='Fraction of training data that is poisoned')
    parser.add_argument('--train_budget', default=0.2, type=float, help='Fraction of clean training data to use for training')
    parser.add_argument('--targets', default=3000, type=int, help='Number of targets')
    
    # Target grouping for focused poisoning (backward compatible - defaults preserve original behavior):
    parser.add_argument('--targets_per_group', default=50, type=int, help='Number of targets per group. If None, all targets in one group (original behavior)')
    parser.add_argument('--allocation_strategy', default='sequential', type=str, choices=['sequential', 'random'], 
                        help='How to allocate poisons to target groups') #
    
    # Target splitting based on original model loss:
    parser.add_argument('--split_targets_by_loss', default=1000, type=int, 
                        help='Split targets into two subsets: top X correctly classified targets with highest original model loss get high poison fraction, rest get low fraction')
    parser.add_argument('--high_loss_poison_fraction', default=0.5, type=float,
                        help='Fraction of poison allocated to high-loss targets, remainder goes to low-loss targets)')

    # Files and folders
    parser.add_argument('--name', default='', type=str, help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--poison_path', default='poisons/', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)
    ###########################################################################

    # Poison brewing:
    parser.add_argument('--attackoptim', default='signAdam', type=str)
    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--init', default='randn', type=str)
    parser.add_argument('--tau', default=0.1, type=float) # Poison step size
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')
    parser.add_argument('--target_criterion', default='untargeted-cross-entropy', type=str, help='Loss criterion for target loss') #untargeted-cross-entropy
    parser.add_argument('--restarts', default=8, type=int, help='How often to restart the attack.')
    parser.add_argument('--reg_factor', default=0.0000001, type=float, help='Regularization factor for poison batch during optimization') # 1
    parser.add_argument('--pbatch', default=500, type=int, help='Poison batch size during optimization')
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    parser.add_argument('--paugment', action='store_false', help='Do not augment poison batch during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    # Poisoning algorithm changes
    parser.add_argument('--full_data', action='store_true', help='Use full train data (instead of just the poison images)')
    parser.add_argument('--adversarial', default=0, type=float, help='Adversarial PGD for poisoning.')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--stagger', action='store_true', help='Stagger the network ensemble if it exists')
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.')
    parser.add_argument('--max_epoch', default=None, type=int, help='Train only up to this epoch before poisoning.')

    # Use only a subset of the dataset:
    parser.add_argument('--ablation', default=1.0, type=float, help='What percent of data (including poisons) to use for validation')

    # Gradient Matching - Specific Options
    parser.add_argument('--loss', default='similarity', type=str)  # similarity is stronger in  difficult situations

    # These are additional regularization terms for gradient matching. We do not use them, but it is possible
    # that scenarios exist in which additional regularization of the poisoned data is useful.
    parser.add_argument('--centreg', default=0, type=float)
    parser.add_argument('--normreg', default=0, type=float)
    parser.add_argument('--repel', default=0, type=float)

    # Specific Options for a metalearning recipe
    parser.add_argument('--nadapt', default=2, type=int, help='Meta unrolling steps')
    parser.add_argument('--clean_grad', action='store_true', help='Compute the first-order poison gradient.')

    # Validation behavior
    parser.add_argument('--vruns', default=0, type=int, help='How often to re-initialize and check target after retraining')
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')], help='Evaluate poison on this victim model. Defaults to --net')
    parser.add_argument('--retrain_from_init', action='store_true', help='Additionally evaluate by retraining on the same model initialization.')
    parser.add_argument('--continue_training_to_loss', action='store_true', help='Continue training poisoned model until it reaches original model training loss')
    parser.add_argument('--duplicate_poison_training', action='store_true', help='During retraining, duplicate poison samples: use both clean and poisoned versions')

    # Optimization setup
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained models from torchvision, if possible [only valid for ImageNet].')
    parser.add_argument('--optimization', default='conservative', type=str, help='Optimization Strategy')
    # Strategy overrides:
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--noaugment', action='store_true', help='Do not use data augmentation during training.')
    parser.add_argument('--gradient_noise', default=None, type=float, help='Add custom gradient noise during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Optionally, datasets can be stored as LMDB or within RAM:
    parser.add_argument('--lmdb_path', default=None, type=str)
    parser.add_argument('--cache_dataset', action='store_true', help='Cache the entire thing :>')

    # These options allow for testing against the toxicity benchmark found at
    # https://github.com/aks2203/poisoning-benchmark
    parser.add_argument('--benchmark', default='', type=str, help='Path to benchmarking setup (pickle file)')
    parser.add_argument('--benchmark_idx', default=0, type=int, help='Index of benchmark test')

    # Debugging:
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--save', default=None, help='Export poisons into a given format. Options are full/limited/automl/numpy.')

    # GPU Control
    parser.add_argument('--gpus', default="auto", type=str, help='Specify which GPUs to use. Examples: "0" (single GPU), "0,1,2" (multiple), "auto" (all available), "cpu" (CPU only)')
    parser.add_argument('--max_gpus', default=None, type=int, help='Limit the maximum number of GPUs to use (uses first N available)')
    
    # Distributed Computations
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')

    return parser
