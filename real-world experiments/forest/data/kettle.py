"""Data class, holding information about dataloaders and poison ids."""

import torch
import numpy as np

import pickle

import datetime
import os
import warnings
import random
import PIL

from .datasets import construct_datasets, Subset
from .cached_dataset import CachedDataset

from .diff_data_augmentation import RandomTransform

from ..consts import PIN_MEMORY, BENCHMARK, DISTRIBUTED_BACKEND, SHARING_STRATEGY, MAX_THREADING
from ..utils import set_random_seed
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class Kettle():
    """Brew poison with given arguments.

    Data class.
    Attributes:
    - trainloader
    - validloader
    - poisonloader
    - poison_ids
    - trainset/poisonset/targetset

    Most notably .poison_lookup is a dictionary that maps image ids to their slice in the poison_delta tensor.

    Initializing this class will set up all necessary attributes.

    Other data-related methods of this class:
    - initialize_poison
    - export_poison

    """

    def __init__(self, args, batch_size, augmentations, setup=dict(device=torch.device('cpu'), dtype=torch.float), include_poison=True): #
        """Initialize with given specs..."""
        self.args, self.setup = args, setup
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.include_poison = include_poison #
        self.trainset, self.validset = self.prepare_data(normalize=True)
        num_workers = self.get_num_workers()

        if self.args.lmdb_path is not None:
            from .lmdb_datasets import LMDBDataset  # this also depends on py-lmdb
            self.trainset = LMDBDataset(self.trainset, self.args.lmdb_path, 'train')
            self.validset = LMDBDataset(self.validset, self.args.lmdb_path, 'val')

        if self.args.cache_dataset:
            self.trainset = CachedDataset(self.trainset, num_workers=num_workers)
            self.validset = CachedDataset(self.validset, num_workers=num_workers)
            num_workers = 0

        if self.args.poisonkey is None:
            if self.args.benchmark != '':
                with open(self.args.benchmark, 'rb') as handle:
                    setup_dict = pickle.load(handle)
                self.benchmark_construction(setup_dict[self.args.benchmark_idx])  # using the first setup dict for benchmarking
            else:
                self.random_construction()


        else:
            if '-' in self.args.poisonkey:
                # If the poisonkey contains a dash-separated triplet like 5-3-1, then poisons are drawn
                # entirely deterministically.
                self.deterministic_construction()
            else:
                # Otherwise the poisoning process is random.
                # If the poisonkey is a random integer, then this integer will be used
                # as a key to seed the random generators.
                self.random_construction()


        # Generate loaders:
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                       shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=min(self.batch_size, len(self.validset)),
                                                       shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        validated_batch_size = max(min(args.pbatch, len(self.poisonset)), 1)
        self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=validated_batch_size,
                                                        shuffle=self.args.pshuffle, drop_last=False, num_workers=num_workers,
                                                        pin_memory=PIN_MEMORY)

        # Ablation on a subset?
        if args.ablation < 1.0:
            self.sample = random.sample(range(len(self.trainset)), int(self.args.ablation * len(self.trainset)))
            self.partialset = Subset(self.trainset, self.sample)
            self.partialloader = torch.utils.data.DataLoader(self.partialset, batch_size=min(self.batch_size, len(self.partialset)),
                                                             shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.print_status()


    """ STATUS METHODS """

    def print_status(self):
        class_names = self.trainset.classes
        print(
            f'Poisoning setup generated for threat model {self.args.threatmodel} and '
            f'budget of {self.args.budget * 100}% - {len(self.poisonset)} images:')
        print(
            f'--Target images drawn from class {", ".join([class_names[self.targetset[i][1]] for i in range(len(self.targetset))])}'
            f' with ids {self.target_ids}.')
        print(f'--Target images assigned intended class {", ".join([class_names[i] for i in self.poison_setup["intended_class"]])}.')

        if self.poison_setup["poison_class"] is not None:
            print(f'--Poison images drawn from class {class_names[self.poison_setup["poison_class"]]}.')
        else:
            print(f'--Poison images drawn from all classes.')

        if self.args.ablation < 1.0:
            print(f'--Partialset is {len(self.partialset)/len(self.trainset):2.2%} of full training set')
            num_p_poisons = len(np.intersect1d(self.poison_ids.cpu().numpy(), np.array(self.sample)))
            print(f'--Poisons in partialset are {num_p_poisons} ({num_p_poisons/len(self.poison_ids):2.2%})')

    def get_num_workers(self):
        """Check devices and set an appropriate number of workers."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_num_workers = 4 * num_gpus
        else:
            max_num_workers = 4
        if torch.get_num_threads() > 1 and MAX_THREADING > 0:
            worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
        else:
            worker_count = 0
        # worker_count = 200
        print(f'Data is loaded with {worker_count} workers.')
        return worker_count

    """ CONSTRUCTION METHODS """

    def prepare_data(self, normalize=True):
        trainset, validset = construct_datasets(self.args.dataset, self.args.data_path, normalize, self.args.modelkey) #


        # Prepare data mean and std for later:
        self.dm = torch.tensor(trainset.data_mean)[None, :, None, None].to(**self.setup)
        self.ds = torch.tensor(trainset.data_std)[None, :, None, None].to(**self.setup)


        # Train augmentations are handled separately as they possibly have to be backpropagated
        if self.augmentations is not None or self.args.paugment:
            if 'CIFAR' in self.args.dataset:
                params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
            elif 'MNIST' in self.args.dataset:
                params = dict(source_size=28, target_size=28, shift=4, fliplr=True)
            elif 'TinyImageNet' in self.args.dataset:
                params = dict(source_size=64, target_size=64, shift=64 // 4, fliplr=True)
            elif 'ImageNet' in self.args.dataset:
                params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)

            if self.augmentations == 'default':
                self.augment = RandomTransform(**params, mode='bilinear')
            elif not self.defs.augmentations:
                print('Data augmentations are disabled.')
                self.augment = RandomTransform(**params, mode='bilinear')
            else:
                raise ValueError(f'Invalid diff. transformation given: {self.augmentations}.')
        return trainset, validset

    def deterministic_construction(self):
        """Construct according to the triplet input key.

        The triplet key, e.g. 5-3-1 denotes in order:
        target_class - poison_class - target_id

        Poisons are always the first n occurences of the given class.
        [This is the same setup as in metapoison]
        """
        if self.args.threatmodel != 'single-class':
            raise NotImplementedError()

        split = self.args.poisonkey.split('-')
        if len(split) != 3:
            raise ValueError('Invalid poison triplet supplied.')
        else:
            target_class, poison_class, target_id = [int(s) for s in split]
        self.init_seed = self.args.poisonkey
        print(f'Initializing Poison data (chosen images, examples, targets, labels) as {self.args.poisonkey}')

        self.poison_setup = dict(poison_budget=self.args.budget,
                                 target_num=self.args.targets, poison_class=poison_class, target_class=target_class,
                                 intended_class=[poison_class])
        self.poisonset, self.targetset, self.validset = self._choose_poisons_deterministic(target_id)

    def benchmark_construction(self, setup_dict):
        """Construct according to the benchmark."""
        target_class, poison_class = setup_dict['target class'], setup_dict['base class']

        budget = len(setup_dict['base indices']) / len(self.trainset)
        self.poison_setup = dict(poison_budget=budget,
                                 target_num=self.args.targets, poison_class=poison_class, target_class=target_class,
                                 intended_class=[poison_class])
        self.init_seed = self.args.poisonkey
        self.poisonset, self.targetset, self.validset = self._choose_poisons_benchmark(setup_dict)

    def _choose_poisons_benchmark(self, setup_dict):
        # poisons
        class_ids = setup_dict['base indices']
        poison_num = len(class_ids)
        self.poison_ids = class_ids

        # the target
        self.target_ids = [setup_dict['target index']]
        # self.target_ids = setup_dict['target index']

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids, range(poison_num)))

        return poisonset, targetset, validset

    def _choose_poisons_deterministic(self, target_id):
        # poisons
        class_ids = []
        for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
            target, idx = self.trainset.get_target(index)
            if target == self.poison_setup['poison_class']:
                class_ids.append(idx)

        poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
        if len(class_ids) < poison_num:
            warnings.warn(f'Training set is too small for requested poison budget.')
            poison_num = len(class_ids)
        self.poison_ids = class_ids[:poison_num]

        # the target
        # class_ids = []
        # for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
        #     target, idx = self.validset.get_target(index)
        #     if target == self.poison_setup['target_class']:
        #         class_ids.append(idx)
        # self.target_ids = [class_ids[target_id]]
        # Disable for now for benchmark sanity check. This is a breaking change.
        self.target_ids = [target_id]

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids, range(poison_num)))
        dict(zip(self.poison_ids, range(poison_num)))
        return poisonset, targetset, validset

    def random_construction(self):
        """Construct according to random selection.

        The setup can be repeated from its key (which initializes the random generator).
        This method sets
         - poison_setup
         - poisonset / targetset / validset

        """
        if self.args.local_rank is None:
            if self.args.poisonkey is None:
                self.init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.init_seed = int(self.args.poisonkey)
            set_random_seed(self.init_seed)
            print(f'Initializing Poison data (chosen images, examples, targets, labels) with random seed {self.init_seed}')
        else:
            rank = torch.distributed.get_rank()
            if self.args.poisonkey is None:
                init_seed = torch.randint(0, 2**32 - 1, [1], device=self.setup['device'])
            else:
                init_seed = torch.as_tensor(int(self.args.poisonkey), dtype=torch.int64, device=self.setup['device'])
            torch.distributed.broadcast(init_seed, src=0)
            if rank == 0:
                print(f'Initializing Poison data (chosen images, examples, targets, labels) with random seed {init_seed.item()}')
            self.init_seed = init_seed.item()
            set_random_seed(self.init_seed)
        # Parse threat model
        self.poison_setup = self._parse_threats_randomly()
        self.poisonset, self.targetset, self.validset = self._choose_poisons_randomly()

    def _parse_threats_randomly(self):
        """Parse the different threat models.

        The threat-models are [In order of expected difficulty]:

        single-class replicates the threat model of feature collision attacks,
        third-party draws all poisons from a class that is unrelated to both target and intended label.
        random-subset draws poison images from all classes.
        random-subset draw poison images from all classes and draws targets from different classes to which it assigns
        different labels.
        """
        num_classes = len(self.trainset.classes)

        target_class = np.random.randint(num_classes)
        list_intentions = list(range(num_classes))
        list_intentions.remove(target_class)
        intended_class = [np.random.choice(list_intentions)] * self.args.targets

        if self.args.targets < 1:
            poison_setup = dict(poison_budget=0, target_num=0,
                                poison_class=np.random.randint(num_classes), target_class=None,
                                intended_class=[np.random.randint(num_classes)])
            warnings.warn('Number of targets set to 0.')
            return poison_setup

        if self.args.threatmodel == 'single-class':
            poison_class = intended_class[0]
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class)
        elif self.args.threatmodel == 'third-party':
            list_intentions.remove(intended_class[0])
            poison_class = np.random.choice(list_intentions)
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class)
        elif self.args.threatmodel == 'self-betrayal':
            poison_class = target_class
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class)
        elif self.args.threatmodel == 'random-subset':
            poison_class = None
            poison_setup = dict(poison_budget=self.args.budget,
                                target_num=self.args.targets, poison_class=None, target_class=target_class,
                                intended_class=intended_class)
        elif self.args.threatmodel == 'random-subset-random-targets':
            target_class = None
            intended_class = np.random.randint(num_classes, size=self.args.targets)
            poison_class = None
            poison_setup = dict(poison_budget=self.args.budget,
                                target_num=self.args.targets, poison_class=None, target_class=None,
                                intended_class=intended_class)
        else:
            raise NotImplementedError('Unknown threat model.')

        return poison_setup

    def _choose_poisons_randomly(self):
        """Subconstruct poison and targets.

        The behavior is different for poisons and targets. We still consider poisons to be part of the original training
        set and load them via trainloader (And then add the adversarial pattern Delta)
        The targets are fully removed from the validation set and returned as a separate dataset, indicating that they
        should not be considered during clean validation using the validloader

        """

        # Poisons:
        if self.poison_setup['poison_class'] is not None:
            class_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                target, idx = self.trainset.get_target(index)
                if target == self.poison_setup['poison_class']:
                    class_ids.append(idx)

            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))

            if len(class_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(class_ids)}')
                poison_num = len(class_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                class_ids, size=poison_num, replace=False), dtype=torch.long)
        else:
            total_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                _, idx = self.trainset.get_target(index)
                total_ids.append(idx)
            
            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))

            if len(total_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(total_ids)}')
                poison_num = len(total_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                total_ids, size=poison_num, replace=False), dtype=torch.long)

        # Targets:
        if self.poison_setup['target_class'] is not None:
            class_ids = []
            for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
                target, idx = self.validset.get_target(index)
                if target == self.poison_setup['target_class']:
                    class_ids.append(idx)
            self.target_ids = np.random.choice(class_ids, size=self.args.targets, replace=False)
        else:
            total_ids = []
            for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
                _, idx = self.validset.get_target(index)
                total_ids.append(idx)
            self.target_ids = np.random.choice(total_ids, size=self.args.targets, replace=False)

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            # if idx not in self.target_ids:
            #     valid_indices.append(idx)
            valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        clean_train_size = int(self.args.train_budget * len(self.trainset)) #40000 # # 45000
        clean_ids = []  #
        for index in range(len(self.trainset)):  #
            _, idx = self.trainset.get_target(index)  #
            if idx not in self.poison_ids: #
                clean_ids.append(idx)  #
        clean_ids = torch.tensor(np.random.choice(clean_ids, size=clean_train_size, replace=False))
        if not self.include_poison: #
            self.train_ids = clean_ids #
        else: #
            self.train_ids = torch.cat((clean_ids, self.poison_ids), dim=0) #
        self.trainset = Subset(self.trainset, indices=self.train_ids)  #

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids.tolist(), range(poison_num)))
        self.train_lookup = dict(zip(self.train_ids.tolist(), range(clean_train_size + poison_num))) #
        return poisonset, targetset, validset

    def create_duplicated_poison_dataset(self):
        """Create a dataset where poison samples appear twice: once clean, once for poisoning.
        
        This modifies the trainset to include:
        - All clean samples (not in poison_ids)
        - All poison samples twice: once as clean, once as poison candidates
        
        Updates poison_lookup to only map the "poison copy" IDs to poison deltas.
        """
        if not hasattr(self, 'original_train_ids'):
            # Store original state for potential restoration
            self.original_train_ids = self.train_ids.clone()
            self.original_poison_lookup = self.poison_lookup.copy()
            self.original_trainset = self.trainset
        
        # Get clean IDs (not in poison set)
        clean_train_size = int(self.args.train_budget * len(self.trainset.dataset if hasattr(self.trainset, 'dataset') else self.trainset))
        clean_ids = []
        
        # Get the base dataset to work with
        base_dataset = self.trainset.dataset if hasattr(self.trainset, 'dataset') else self.trainset
        
        for index in range(len(base_dataset)):
            _, idx = base_dataset.get_target(index)
            if idx not in self.poison_ids:
                clean_ids.append(idx)
        
        # Sample clean IDs according to train_budget
        clean_ids = torch.tensor(np.random.choice(clean_ids, size=clean_train_size, replace=False))
        
        # Create new IDs for duplicated poison samples
        # We'll use a large offset to ensure no collision with existing IDs
        max_id = max(len(base_dataset), max(self.poison_ids).item() + 1)
        poison_clean_ids = self.poison_ids.clone()  # These will be treated as clean
        poison_dirty_ids = self.poison_ids + max_id  # These will have poison applied
        
        # Combine all IDs: clean + poison_clean + poison_dirty
        self.train_ids = torch.cat((clean_ids, poison_clean_ids, poison_dirty_ids), dim=0)
        
        # Create new poison_lookup that only maps the "dirty" poison IDs
        poison_num = len(self.poison_ids)
        self.poison_lookup = dict(zip(poison_dirty_ids.tolist(), range(poison_num)))
        
        # Update train_lookup
        total_size = len(clean_ids) + len(poison_clean_ids) + len(poison_dirty_ids)
        self.train_lookup = dict(zip(self.train_ids.tolist(), range(total_size)))
        
        # Create a custom dataset that handles the duplicated poison samples
        from .duplicated_dataset import DuplicatedPoisonDataset
        self.trainset = DuplicatedPoisonDataset(
            base_dataset, 
            self.train_ids, 
            self.poison_ids, 
            poison_dirty_ids, 
            max_id
        )
        
        # Update the trainloader
        num_workers = self.get_num_workers()
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=min(self.batch_size, len(self.trainset)),
            shuffle=True, 
            drop_last=False, 
            num_workers=num_workers, 
            pin_memory=PIN_MEMORY
        )
        
        print(f'Created duplicated poison dataset:')
        print(f'  Clean samples: {len(clean_ids)}')
        print(f'  Poison samples (clean copy): {len(poison_clean_ids)}')
        print(f'  Poison samples (poison copy): {len(poison_dirty_ids)}')
        print(f'  Total training samples: {len(self.trainset)}')

    def initialize_poison(self, initializer=None):
        """Initialize according to args.init.

        Propagate initialization in distributed settings.
        """
        if initializer is None:
            initializer = self.args.init

        # ds has to be placed on the default (cpu) device, not like self.ds
        ds = torch.tensor(self.trainset.data_std)[None, :, None, None]
        if initializer == 'zero':
            init = torch.zeros(len(self.poison_ids), *self.trainset[0][0].shape)
        elif initializer == 'rand':
            init = (torch.rand(len(self.poison_ids), *self.trainset[0][0].shape) - 0.5) * 2
            init *= self.args.eps / ds / 255
        elif initializer == 'randn':
            init = torch.randn(len(self.poison_ids), *self.trainset[0][0].shape)
            init *= self.args.eps / ds / 255
        elif initializer == 'normal':
            init = torch.randn(len(self.poison_ids), *self.trainset[0][0].shape)
        else:
            raise NotImplementedError()

        init.data = torch.max(torch.min(init, self.args.eps / ds / 255), -self.args.eps / ds / 255)

        # If distributed, sync poison initializations
        if self.args.local_rank is not None:
            if DISTRIBUTED_BACKEND == 'nccl':
                init = init.to(device=self.setup['device'])
                torch.distributed.broadcast(init, src=0)
                init.to(device=torch.device('cpu'))
            else:
                torch.distributed.broadcast(init, src=0)
        return init

    """ EXPORT METHODS """

    def export_poison(self, poison_delta, path=None, mode='automl', model_seed=None):
        """Export poisons in either packed mode (just ids and raw data) or in full export mode, exporting all images.

        In full export mode, export data into folder structure that can be read by a torchvision.datasets.ImageFolder

        In automl export mode, export data into a single folder and produce a csv file that can be uploaded to
        google storage.
        """
        if path is None:
            path = self.args.poison_path

        dm = torch.tensor(self.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            if image_torch_uint8.shape[-1] == 1: #
                image_torch_uint8 = image_torch_uint8.squeeze(-1) #
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _save_image(input, label, idx, location, train=True):
            """Save input image to given location, add poison_delta if necessary."""
            filename = os.path.join(location, str(idx) + '.png')

            lookup = self.poison_lookup.get(idx)
            if (lookup is not None) and train:
                input += poison_delta[lookup, :, :, :]
            _torch_to_PIL(input).save(filename)

        # Save either into packed mode, ImageDataSet Mode or google storage mode
        if mode == 'packed':
            data = dict()
            data['poison_setup'] = self.poison_setup
            data['poison_delta'] = poison_delta
            data['poison_ids'] = self.poison_ids
            data['target_images'] = [data for data in self.targetset]
            name = f'{path}poisons_packed_{datetime.date.today()}.pth'
            torch.save([poison_delta, self.poison_ids], os.path.join(path, name))

        elif mode == 'limited':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.targetset):
                intended_class = self.poison_setup['intended_class'][enum]
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)
            print('Target images exported with intended class labels ...')

        elif mode == 'full':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
            for input, label, idx in self.trainset:
                _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')

            for input, label, idx in self.validset:
                _save_image(input, label, idx, location=os.path.join(path, 'test', names[label]), train=False)
            print('Unaffected validation images exported ...')

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.targetset):
                intended_class = self.poison_setup['intended_class'][enum]
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)
            print('Target images exported with intended class labels ...')

        elif mode in ['automl-upload', 'automl-all', 'automl-baseline']:
            from ..utils import automl_bridge
            targetclass = self.targetset[0][1]
            poisonclass = self.poison_setup["poison_class"]

            name_candidate = f'{self.args.name}_{self.args.dataset}T{targetclass}P{poisonclass}'
            name = ''.join(e for e in name_candidate if e.isalnum())

            if mode == 'automl-upload':
                automl_phase = 'poison-upload'
            elif mode == 'automl-all':
                automl_phase = 'all'
            elif mode == 'automl-baseline':
                automl_phase = 'upload'
            automl_bridge(self, poison_delta, name, mode=automl_phase, dryrun=self.args.dryrun)

        elif mode == 'numpy':
            #_, h, w = self.trainset[0][0].shape
            #np.zeros([len(self.trainset), h, w, 3])
            c, h, w = self.trainset[0][0].shape #
            if c == 1: #
                training_data = np.zeros([len(self.trainset), h, w]) #
            else: #
                training_data = np.zeros([len(self.trainset), h, w, c]) #
            labels = np.zeros(len(self.trainset))
            deltas = [] #
            poison_ids = []
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx.item()) #
                if lookup is not None:
                    input += poison_delta[lookup, :, :, :]
                    deltas.append(poison_delta[lookup, :, :, :].cpu().numpy()) #
                    poison_ids.append(idx.item())
                training_data[self.train_lookup[idx.item()]] = np.asarray(_torch_to_PIL(input)) #
                labels[self.train_lookup[idx.item()]] = label #
            deltas = np.asarray(deltas) #
            
            # Compute last row norm ratio (percentage of norm in last row vs entire image)
            N = deltas.shape[0]
            if N > 0 and len(deltas.shape) == 4:  # Ensure we have poison deltas with shape (N, C, H, W)
                # deltas shape: (N, C, H, W)
                last_row_norms = np.linalg.norm(deltas[:, :, -1, :].reshape(N, -1), axis=1)
                image_norms = np.linalg.norm(deltas.reshape(N, -1), axis=1)
                # Avoid division by zero
                ratios = np.divide(last_row_norms, image_norms, out=np.zeros_like(last_row_norms), where=image_norms!=0)
                last_row_norm_ratio = np.mean(ratios)
            else:
                last_row_norm_ratio = 0.0
            
            # Create filename with seeds included
            base_name = self.args.name
            if model_seed is not None:
                filename_prefix = f'{base_name}_model_seed={model_seed}_poison_seed={self.init_seed}'
            else:
                filename_prefix = f'{base_name}_poison_seed={self.init_seed}'
            
            np.save(os.path.join(path, f'{filename_prefix}_deltas.npy'), deltas) #
            np.save(os.path.join(path, f'{filename_prefix}_poisoned_training_data.npy'), training_data)
            np.save(os.path.join(path, f'{filename_prefix}_poisoned_training_labels.npy'), labels)
            np.save(os.path.join(path, f'{filename_prefix}_poison_ids.npy'), poison_ids)
            
            return last_row_norm_ratio
            
        elif mode == 'kettle-export':
            with open(f'kette_{self.args.dataset}{self.args.model}.pkl', 'wb') as file:
                pickle.dump([self, poison_delta], file, protocol=pickle.HIGHEST_PROTOCOL)

        elif mode == 'benchmark':
            foldername = f'{self.args.name}_{"_".join(self.args.net)}'
            sub_path = os.path.join(path, 'benchmark_results', foldername, str(self.args.benchmark_idx))
            os.makedirs(sub_path, exist_ok=True)

            # Poisons
            benchmark_poisons = []
            for lookup, key in enumerate(self.poison_lookup.keys()):  # This is a different order than we usually do for compatibility with the benchmark
                input, label, _ = self.trainset[key]
                input += poison_delta[lookup, :, :, :]
                benchmark_poisons.append((_torch_to_PIL(input), int(label)))

            with open(os.path.join(sub_path, 'poisons.pickle'), 'wb+') as file:
                pickle.dump(benchmark_poisons, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Target
            target, target_label, _ = self.targetset[0]
            with open(os.path.join(sub_path, 'target.pickle'), 'wb+') as file:
                pickle.dump((_torch_to_PIL(target), target_label), file, protocol=pickle.HIGHEST_PROTOCOL)

            # Indices
            with open(os.path.join(sub_path, 'base_indices.pickle'), 'wb+') as file:
                pickle.dump(self.poison_ids, file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            raise NotImplementedError()

        print('Dataset fully exported.')
        return None  # Return None for non-numpy modes
