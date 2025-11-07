"""General interface script to launch poisoning jobs."""

import os #
os.environ['OMP_NUM_THREADS'] = '8' #
os.environ['MKL_NUM_THREADS'] = '8' # 

import torch
torch.set_num_threads(8) #

import datetime
import time

import forest
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()

# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()


if __name__ == "__main__":

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup) #
    #data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup, include_poison=False) # Add poison
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup, include_poison=True) # Replace poison
    witch = forest.Witch(args, setup=setup) #

    start_time = time.time()
    
    if args.pretrained:
        print('Loading pretrained model...')
        stats_clean = None
        poison_delta = None  # Will be set later
    else:
        stats_clean = model.train(data, max_epoch=args.max_epoch) #
        poison_delta = None  # Will be set later
    
    train_time = time.time()

    # Reset data object to include poison
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup, include_poison=True) #
    
    # Brew poison
    poison_delta = witch.brew(model, data) #
    brew_time = time.time()

    if not args.pretrained and args.retrain_from_init:
        # Use duplicated poison dataset if requested
        if args.duplicate_poison_training:
            print('Using duplicated poison training: poison samples appear both clean and poisoned...')
            data.create_duplicated_poison_dataset()
        
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None  # we dont know the initial seed for a pretrained model so retraining makes no sense

    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)
            # Use duplicated poison dataset for validation if requested
            if args.duplicate_poison_training:
                data.create_duplicated_poison_dataset()
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
        args.net = train_net
    else:  # Validate the main model
        if args.vruns > 0:
            # Use duplicated poison dataset for validation if requested
            if args.duplicate_poison_training:
                data.create_duplicated_poison_dataset()
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
    test_time = time.time()


    timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                      brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                      test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''))
    
    # Export first if numpy mode to compute last row norm ratio
    last_row_norm_ratio = None
    if args.save is not None:
        last_row_norm_ratio = data.export_poison(poison_delta, path=args.poison_path, mode=args.save, model_seed=model.model_init_seed)
    
    # Add last row norm ratio to extra stats if available (from numpy export)
    if last_row_norm_ratio is not None:
        timestamps['last_row_norm_ratio'] = f'{last_row_norm_ratio:.4f}'
    
    # Save run to table
    results = (stats_clean, stats_rerun, stats_results)
    forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                args, model.defs, model.model_init_seed, extra_stats=timestamps)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
    print('-------------Job finished.-------------------------')
