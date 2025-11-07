"""Single model default victim class."""

import torch
import numpy as np
from collections import defaultdict


from ..utils import set_random_seed
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase


class _VictimSingle(_VictimBase):
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation.

    """

    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.criterion, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0])

        self.model.to(**self.setup)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking target accuracy."""
        stats = defaultdict(list)

        if max_epoch is None:
            max_epoch = self.defs.epochs

        def loss_fn(model, outputs, labels):
            return self.criterion(outputs, labels)

        # Create extended scheduler if continue training feature is enabled
        if self.args.continue_training_to_loss and poison_delta is not None:
            max_additional_epochs = max_epoch * 2  # Safety limit
            from .training import create_extended_scheduler
            extended_scheduler = create_extended_scheduler(self.optimizer, self.defs, max_epoch, max_additional_epochs)
            single_setup = (self.model, self.defs, self.criterion, self.optimizer, extended_scheduler)
        else:
            single_setup = (self.model, self.defs, self.criterion, self.optimizer, self.scheduler)
        
        # Train for the original number of epochs
        for self.epoch in range(max_epoch):
            self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup, max_epoch=max_epoch)
            if self.args.dryrun:
                break
        
        # If we have poison, the feature is enabled, and an original training loss target, continue training until we reach it
        if (poison_delta is not None and self.args.continue_training_to_loss and 
            hasattr(self, 'original_train_loss') and self.original_train_loss is not None and 
            len(stats['train_losses']) > 0):
            
            current_loss = stats['train_losses'][-1]
            print(f'After {max_epoch} epochs: current loss = {current_loss:.6f}, target loss = {self.original_train_loss:.6f}')
            
            # Continue training until we reach the original training loss (with some tolerance)
            tolerance = 0.001  # Allow small tolerance for convergence
            max_additional_epochs = max_epoch * 2  # Safety limit to prevent infinite training
            additional_epoch = 0
            
            while (current_loss > self.original_train_loss + tolerance and 
                   additional_epoch < max_additional_epochs):
                
                self.epoch = max_epoch + additional_epoch
                # Continue with extended scheduler
                self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup, max_epoch=max_epoch)
                
                if len(stats['train_losses']) > 0:
                    current_loss = stats['train_losses'][-1]
                
                additional_epoch += 1
                
                if additional_epoch % 10 == 0:  # Print progress every 10 epochs
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f'Extended training epoch {self.epoch}: current loss = {current_loss:.6f}, target = {self.original_train_loss:.6f}, LR = {current_lr:.8f}')
                
                if self.args.dryrun:
                    break
            
            if additional_epoch > 0:
                total_epochs = max_epoch + additional_epoch
                final_lr = self.optimizer.param_groups[0]['lr']
                print(f'Extended training completed after {additional_epoch} additional epochs (total: {total_epochs})')
                print(f'Final loss: {current_loss:.6f}, target: {self.original_train_loss:.6f}, final LR: {final_lr:.8f}')
        
        return stats

    def step(self, kettle, poison_delta, poison_targets, true_classes):
        """Step through a model epoch. Optionally: minimize target loss."""
        stats = defaultdict(list)

        def loss_fn(model, outputs, labels):
            normal_loss = self.criterion(outputs, labels)
            model.eval()
            if self.args.adversarial != 0:
                target_loss = 1 / self.defs.batch_size * self.criterion(model(poison_targets), true_classes)
            else:
                target_loss = 0
            model.train()
            return normal_loss + self.args.adversarial * target_loss

        single_setup = (self.model, self.criterion, self.optimizer, self.scheduler)
        self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs:
            self.epoch = 0
            print('Model reset to epoch 0.')
            self.model, self.criterion, self.optimizer, self.scheduler = self._initialize_model()
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        return stats

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler = self._initialize_model()

    def gradient(self, images, labels, criterion=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        if criterion is None:
            loss = self.criterion(self.model(images), labels)
        else:
            loss = criterion(self.model(images), labels)
        gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.criterion, self.optimizer, *args)
