"""Main class, holding information about models and training/testing routines."""

import time
import random

import torch
import warnings

from ..utils import cw_loss
from ..consts import NON_BLOCKING, BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

class _Witch():
    """Brew poison with given arguments.

    Base class.

    This class implements _brew(), which is the main loop for iterative poisoning.
    New iterative poisoning methods overwrite the _define_objective method.

    Noniterative poison methods overwrite the _brew() method itself.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        self.retain = True if self.args.ensemble > 1 and self.args.local_rank is None else False
        self.stat_optimal_loss = None

    """ BREWING RECIPES """

    def brew(self, victim, kettle):
        """Recipe interface."""
        if len(kettle.poisonset) > 0:
            if len(kettle.targetset) > 0:
                if self.args.eps > 0:
                    if self.args.budget > 0:
                        poison_delta = self._brew(victim, kettle)
                    else:
                        poison_delta = kettle.initialize_poison(initializer='zero')
                        warnings.warn('No poison budget given. Nothing can be poisoned.')
                else:
                    poison_delta = kettle.initialize_poison(initializer='zero')
                    warnings.warn('Perturbation interval is empty. Nothing can be poisoned.')
            else:
                poison_delta = kettle.initialize_poison(initializer='zero')
                warnings.warn('Target set is empty. Nothing can be poisoned.')
        else:
            poison_delta = kettle.initialize_poison(initializer='zero')
            warnings.warn('Poison set is empty. Nothing can be poisoned.')

        return poison_delta

    def _brew(self, victim, kettle):
        """Run generalized iterative routine."""
        print(f'Starting brewing procedure ...')
        self._initialize_brew(victim, kettle)
        poisons, scores = [], torch.ones(self.args.restarts) * 10_000

        for trial in range(self.args.restarts):
            poison_delta, target_losses = self._run_trial(victim, kettle)
            scores[trial] = target_losses
            poisons.append(poison_delta.detach())
            if self.args.dryrun:
                break

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        print(f'Poisons with minimal target loss {self.stat_optimal_loss:6.4e} selected.')
        poison_delta = poisons[optimal_score]

        return poison_delta


    def _initialize_brew(self, victim, kettle):
        """Implement common initialization operations for brewing."""
        victim.eval(dropout=True)
        # Compute target gradients
        self.targets = torch.stack([data[0] for data in kettle.targetset], dim=0).to(**self.setup)
        self.intended_classes = torch.tensor(kettle.poison_setup['intended_class']).to(device=self.setup['device'], dtype=torch.long)
        self.true_classes = torch.tensor([data[1] for data in kettle.targetset]).to(device=self.setup['device'], dtype=torch.long)

        # Set up target grouping (backward compatible)
        self._setup_target_groups(len(kettle.targetset))
        
        # Reorder targets by loss if loss-based splitting is enabled
        if hasattr(self, 'target_split_enabled') and self.target_split_enabled:
            self._reorder_targets_by_loss(victim, kettle)
        
        self._setup_poison_to_group_mapping(len(kettle.poisonset))

        # Compute target gradients
        if self.num_groups == 1:
            # Original behavior: compute gradients for all targets at once
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.intended_classes, cw_loss)
            elif self.args.target_criterion in ['untargeted-cross-entropy', 'unxent']:
                self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.true_classes)
                for grad in self.target_grad:
                    grad *= -1
            elif self.args.target_criterion in ['xent', 'cross-entropy']:
                self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.intended_classes)
            else:
                raise ValueError('Invalid target criterion chosen ...')
        else:
            # Focused attack: compute gradients separately for each target group
            self._compute_focused_target_gradients(victim)
        
        # Handle repel gradients
        if self.args.repel != 0:
            self.target_clean_grad, _ = victim.gradient(self.targets, self.true_classes)
        else:
            self.target_clean_grad = None

        # Print summary
        if self.num_groups == 1:
            print(f'Target Grad Norm is {self.target_gnorm}')
        else:
            avg_gnorm = sum(g["gnorm"] for g in self.target_grad_groups.values()) / self.num_groups
            print(f'Focused attack: {self.num_groups} groups with {self.targets_per_group} targets per group')
            print(f'Average Target Grad Norm: {avg_gnorm:.6f}')

        # The PGD tau that will actually be used:
        # This is not super-relevant for the adam variants
        # but the PGD variants are especially sensitive
        # E.G: 92% for PGD with rule 1 and 20% for rule 2
        if self.args.attackoptim in ['PGD', 'GD']:
            # Rule 1
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
        elif self.args.attackoptim in ['momSGD', 'momPGD']:
            # Rule 1a
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
            self.tau0 = self.tau0.mean()
        else:
            # Rule 2
            self.tau0 = self.args.tau * (self.args.pbatch / 512) / self.args.ensemble

    def _setup_target_groups(self, num_targets):
        """Set up target grouping configuration (backward compatible)."""
        # Check if we need to split targets by loss first
        if self.args.split_targets_by_loss is not None:
            self._setup_loss_based_target_splitting(num_targets)
        elif self.args.targets_per_group is None:
            # Original behavior: all targets in one group
            self.targets_per_group = num_targets
            self.num_groups = 1
        else:
            self.targets_per_group = self.args.targets_per_group
            self.num_groups = (num_targets + self.targets_per_group - 1) // self.targets_per_group
        
        print(f'Target grouping: {self.num_groups} groups with {self.targets_per_group} targets per group')
    
    def _setup_loss_based_target_splitting(self, num_targets):
        """Split targets into high-loss and low-loss groups with different poison allocations."""
        # This method will be called during brewing when we have access to the victim model
        # For now, set up the basic structure assuming we'll reorder targets later
        high_loss_count = min(self.args.split_targets_by_loss, num_targets)
        low_loss_count = num_targets - high_loss_count
        
        if high_loss_count == 0 or low_loss_count == 0:
            print(f'Warning: Invalid target split - using all targets in single group')
            self.targets_per_group = num_targets
            self.num_groups = 1
            self.target_split_enabled = False
            return
        
        # Create two groups: high-loss and low-loss targets
        self.num_groups = 2
        self.high_loss_count = high_loss_count
        self.low_loss_count = low_loss_count
        self.target_split_enabled = True
        
        # Set up group sizes (we'll handle targets_per_group differently for split targets)
        if self.args.targets_per_group is not None:
            # If user also specified targets_per_group, subdivide each loss group
            high_loss_subgroups = (high_loss_count + self.args.targets_per_group - 1) // self.args.targets_per_group
            low_loss_subgroups = (low_loss_count + self.args.targets_per_group - 1) // self.args.targets_per_group
            self.num_groups = high_loss_subgroups + low_loss_subgroups
            self.targets_per_group = self.args.targets_per_group
            print(f'Loss-based splitting with subgroups: {high_loss_subgroups} high-loss groups + {low_loss_subgroups} low-loss groups')
        else:
            # Simple two-group split
            self.targets_per_group = max(high_loss_count, low_loss_count)  # For compatibility
            print(f'Loss-based splitting: {high_loss_count} high-loss targets + {low_loss_count} low-loss targets')

    def _setup_poison_to_group_mapping(self, num_poisons):
        """Create mapping from poison indices to target group indices."""
        self.poison_to_group = {}
        
        if self.num_groups == 1:
            # Original behavior: all poisons target the single group
            for poison_idx in range(num_poisons):
                self.poison_to_group[poison_idx] = 0
        elif hasattr(self, 'target_split_enabled') and self.target_split_enabled:
            # Special handling for loss-based target splitting
            self._setup_loss_based_poison_allocation(num_poisons)
        else:
            # Distribute poisons across groups
            poisons_per_group = num_poisons // self.num_groups
            remaining_poisons = num_poisons % self.num_groups
            
            poison_idx = 0
            for group_id in range(self.num_groups):
                # Some groups get one extra poison if there's a remainder
                group_size = poisons_per_group + (1 if group_id < remaining_poisons else 0)
                
                # Sequential allocation
                for _ in range(group_size):
                    self.poison_to_group[poison_idx] = group_id
                    poison_idx += 1
            
            # If random allocation, shuffle the assignments
            if self.args.allocation_strategy == 'random':
                poison_indices = list(range(num_poisons))
                group_assignments = list(self.poison_to_group.values())
                random.shuffle(group_assignments)
                self.poison_to_group = dict(zip(poison_indices, group_assignments))
        
        print(f'Poison allocation: {[sum(1 for g in self.poison_to_group.values() if g == group_id) for group_id in range(self.num_groups)]} poisons per group')
    
    def _setup_loss_based_poison_allocation(self, num_poisons):
        """Allocate poisons between high-loss and low-loss target groups with specified fractions."""
        high_loss_fraction = self.args.high_loss_poison_fraction
        low_loss_fraction = 1.0 - high_loss_fraction
        
        # Calculate poison allocation
        high_loss_poisons = int(num_poisons * high_loss_fraction)
        low_loss_poisons = num_poisons - high_loss_poisons
        
        print(f'Loss-based poison allocation: {high_loss_poisons} poisons ({high_loss_fraction:.1%}) for high-loss targets, '
              f'{low_loss_poisons} poisons ({low_loss_fraction:.1%}) for low-loss targets')
        
        if self.num_groups == 2:
            # Simple two-group case
            poison_idx = 0
            
            # Allocate to high-loss group (group 0)
            for _ in range(high_loss_poisons):
                self.poison_to_group[poison_idx] = 0
                poison_idx += 1
            
            # Allocate to low-loss group (group 1)  
            for _ in range(low_loss_poisons):
                self.poison_to_group[poison_idx] = 1
                poison_idx += 1
        else:
            # Handle subgroups within high-loss and low-loss categories
            high_loss_subgroups = (self.high_loss_count + self.args.targets_per_group - 1) // self.args.targets_per_group
            low_loss_subgroups = self.num_groups - high_loss_subgroups
            
            # Distribute high-loss poisons among high-loss subgroups
            high_loss_poisons_per_subgroup = high_loss_poisons // high_loss_subgroups
            high_loss_remaining = high_loss_poisons % high_loss_subgroups
            
            # Distribute low-loss poisons among low-loss subgroups
            low_loss_poisons_per_subgroup = low_loss_poisons // low_loss_subgroups
            low_loss_remaining = low_loss_poisons % low_loss_subgroups
            
            poison_idx = 0
            
            # Allocate to high-loss subgroups (groups 0 to high_loss_subgroups-1)
            for group_id in range(high_loss_subgroups):
                group_size = high_loss_poisons_per_subgroup + (1 if group_id < high_loss_remaining else 0)
                for _ in range(group_size):
                    self.poison_to_group[poison_idx] = group_id
                    poison_idx += 1
            
            # Allocate to low-loss subgroups (remaining groups)
            for i, group_id in enumerate(range(high_loss_subgroups, self.num_groups)):
                group_size = low_loss_poisons_per_subgroup + (1 if i < low_loss_remaining else 0)
                for _ in range(group_size):
                    self.poison_to_group[poison_idx] = group_id
                    poison_idx += 1
    
    def _reorder_targets_by_loss(self, victim, kettle):
        """Reorder targets based on their loss values, filtering to correctly classified targets first."""
        print('Evaluating target losses for loss-based splitting...')
        
        # Evaluate losses and predictions for all targets
        target_data = []  # (index, loss, is_correct)
        victim.eval()
        
        with torch.no_grad():
            for i in range(len(self.targets)):
                target_img = self.targets[i:i+1]  # Single target as batch
                true_label = self.true_classes[i:i+1]
                
                # Get model output and compute loss and prediction
                output = victim.model(target_img)
                loss = victim.criterion(output, true_label).item()
                prediction = torch.argmax(output, dim=1)
                is_correct = (prediction == true_label).item()
                
                target_data.append((i, loss, is_correct))
        
        # Filter to only correctly classified targets
        correct_targets = [(i, loss) for i, loss, is_correct in target_data if is_correct]
        incorrect_targets = [(i, loss) for i, loss, is_correct in target_data if not is_correct]
        
        print(f'Found {len(correct_targets)} correctly classified targets and {len(incorrect_targets)} incorrectly classified targets')
        
        if len(correct_targets) < self.args.split_targets_by_loss:
            print(f'Warning: Only {len(correct_targets)} correctly classified targets available, '
                  f'but {self.args.split_targets_by_loss} requested for high-loss group.')
            print(f'Using all {len(correct_targets)} correct targets for high-loss group.')
            actual_high_loss_count = len(correct_targets)
        else:
            actual_high_loss_count = self.args.split_targets_by_loss
        
        # Sort correctly classified targets by loss (highest first)
        correct_targets.sort(key=lambda x: x[1], reverse=True)
        
        # Sort incorrectly classified targets by loss (highest first) 
        incorrect_targets.sort(key=lambda x: x[1], reverse=True)
        
        # Create high-loss group: top X correctly classified targets with highest loss
        high_loss_targets = correct_targets[:actual_high_loss_count]
        
        # Create low-loss group: remaining correctly classified + all incorrectly classified
        remaining_correct = correct_targets[actual_high_loss_count:]
        low_loss_targets = remaining_correct + incorrect_targets
        
        # Update the counts
        self.high_loss_count = len(high_loss_targets)
        self.low_loss_count = len(low_loss_targets)
        
        # Create final reordering: high-loss first, then low-loss
        reorder_indices = [item[0] for item in high_loss_targets] + [item[0] for item in low_loss_targets]
        
        # Reorder targets, intended classes, and true classes
        self.targets = self.targets[reorder_indices]
        self.intended_classes = self.intended_classes[reorder_indices]
        self.true_classes = self.true_classes[reorder_indices]
        
        # Update kettle's poison setup to match reordered targets
        new_intended_classes = [kettle.poison_setup['intended_class'][i] for i in reorder_indices]
        kettle.poison_setup['intended_class'] = new_intended_classes
        
        # Print detailed information about the split
        if len(high_loss_targets) > 0:
            high_loss_losses = [item[1] for item in high_loss_targets]
            print(f'High-loss targets ({self.high_loss_count}): correctly classified with loss range {min(high_loss_losses):.4f} - {max(high_loss_losses):.4f}')
        
        if len(low_loss_targets) > 0:
            low_loss_losses = [item[1] for item in low_loss_targets]
            remaining_correct_count = len(remaining_correct)
            incorrect_count = len(incorrect_targets)
            print(f'Low-loss targets ({self.low_loss_count}): {remaining_correct_count} remaining correct + {incorrect_count} incorrect, loss range {min(low_loss_losses):.4f} - {max(low_loss_losses):.4f}')
        
        print(f'Target reordering complete: {self.high_loss_count} high-loss (correct) + {self.low_loss_count} low-loss targets')

    def _compute_focused_target_gradients(self, victim):
        """Compute target gradients separately for each group for focused attacks."""
        self.target_grad_groups = {}
        
        for group_id in range(self.num_groups):
            # Handle loss-based splitting differently
            if hasattr(self, 'target_split_enabled') and self.target_split_enabled:
                if self.num_groups == 2:
                    # Simple two-group case: high-loss (group 0) and low-loss (group 1)
                    if group_id == 0:
                        start_idx = 0
                        end_idx = self.high_loss_count
                    else:  # group_id == 1
                        start_idx = self.high_loss_count
                        end_idx = len(self.targets)
                else:
                    # Subgroups within high-loss and low-loss categories
                    high_loss_subgroups = (self.high_loss_count + self.args.targets_per_group - 1) // self.args.targets_per_group
                    
                    if group_id < high_loss_subgroups:
                        # High-loss subgroup
                        start_idx = group_id * self.args.targets_per_group
                        end_idx = min((group_id + 1) * self.args.targets_per_group, self.high_loss_count)
                    else:
                        # Low-loss subgroup
                        low_loss_group_idx = group_id - high_loss_subgroups
                        start_idx = self.high_loss_count + low_loss_group_idx * self.args.targets_per_group
                        end_idx = min(self.high_loss_count + (low_loss_group_idx + 1) * self.args.targets_per_group, len(self.targets))
            else:
                # Original grouping behavior
                start_idx = group_id * self.targets_per_group
                end_idx = min((group_id + 1) * self.targets_per_group, len(self.targets))
            
            # Get targets for this group
            group_targets = self.targets[start_idx:end_idx]
            group_intended_classes = self.intended_classes[start_idx:end_idx]
            group_true_classes = self.true_classes[start_idx:end_idx]
            
            # Compute gradients for this group
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                group_grad, group_gnorm = victim.gradient(group_targets, group_intended_classes, cw_loss)
            elif self.args.target_criterion in ['untargeted-cross-entropy', 'unxent']:
                group_grad, group_gnorm = victim.gradient(group_targets, group_true_classes)
                # Apply negation for untargeted attacks
                group_grad = [grad * -1 for grad in group_grad]
            elif self.args.target_criterion in ['xent', 'cross-entropy']:
                group_grad, group_gnorm = victim.gradient(group_targets, group_intended_classes)
            else:
                raise ValueError('Invalid target criterion chosen ...')
            
            self.target_grad_groups[group_id] = {
                'grad': group_grad,
                'gnorm': group_gnorm,
                'target_indices': list(range(start_idx, end_idx))
            }
        
        # Also compute clean gradients by group if needed
        if self.args.repel != 0:
            self.target_clean_grad_groups = {}
            for group_id in range(self.num_groups):
                # Use the same indexing logic as above
                if hasattr(self, 'target_split_enabled') and self.target_split_enabled:
                    if self.num_groups == 2:
                        if group_id == 0:
                            start_idx = 0
                            end_idx = self.high_loss_count
                        else:
                            start_idx = self.high_loss_count
                            end_idx = len(self.targets)
                    else:
                        high_loss_subgroups = (self.high_loss_count + self.args.targets_per_group - 1) // self.args.targets_per_group
                        
                        if group_id < high_loss_subgroups:
                            start_idx = group_id * self.args.targets_per_group
                            end_idx = min((group_id + 1) * self.args.targets_per_group, self.high_loss_count)
                        else:
                            low_loss_group_idx = group_id - high_loss_subgroups
                            start_idx = self.high_loss_count + low_loss_group_idx * self.args.targets_per_group
                            end_idx = min(self.high_loss_count + (low_loss_group_idx + 1) * self.args.targets_per_group, len(self.targets))
                else:
                    start_idx = group_id * self.targets_per_group
                    end_idx = min((group_id + 1) * self.targets_per_group, len(self.targets))
                
                group_targets = self.targets[start_idx:end_idx]
                group_true_classes = self.true_classes[start_idx:end_idx]
                
                group_clean_grad, _ = victim.gradient(group_targets, group_true_classes)
                self.target_clean_grad_groups[group_id] = group_clean_grad

    def _get_target_grad_for_batch(self, poison_slices):
        """Get appropriate target gradients for a batch of poisons."""
        # This method is only used for single-group (backward compatible) case
        # For focused attacks, gradients are handled in _process_group_subset
        if self.num_groups == 1:
            return self.target_grad, self.target_clean_grad, self.target_gnorm
        else:
            # This shouldn't be called for focused attacks, but just in case
            raise RuntimeError("_get_target_grad_for_batch should not be called for focused attacks")


    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        poison_delta = kettle.initialize_poison()
        if self.args.full_data:
            dataloader = kettle.trainloader
        else:
            dataloader = kettle.poisonloader

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            # poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
            else:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            if self.args.scheduling:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
            poison_delta.grad = torch.zeros_like(poison_delta)
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            poison_bounds = torch.zeros_like(poison_delta)
        else:
            poison_bounds = None

        start_time = time.time()

        for step in range(self.args.attackiter):
            target_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, victim, kettle)
                target_losses += loss
                poison_correct += prediction

                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                if self.args.attackoptim in ['momPGD', 'signAdam']:
                    poison_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad(set_to_none=False)
                with torch.no_grad():
                    # Projection Step
                    poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps /
                                                            ds / 255), -self.args.eps / ds / 255)
                    poison_delta.data = torch.max(torch.min(poison_delta, (1 - dm) / ds -
                                                            poison_bounds), -dm / ds - poison_bounds)

            target_losses = target_losses / (batch + 1)
            poison_acc = poison_correct / len(dataloader.dataset)
            if step % (self.args.attackiter // 5) == 0 or step == (self.args.attackiter - 1):
                print(f'Iteration {step}: Target loss is {target_losses:2.4f}, '
                      f'Poison clean acc is {poison_acc * 100:2.2f}%')
                print(f'Time taken: {time.time() - start_time:.2f} seconds')
                start_time = time.time()

            if self.args.step:
                if self.args.clean_grad:
                    victim.step(kettle, None, self.targets, self.true_classes)
                else:
                    victim.step(kettle, poison_delta, self.targets, self.true_classes)

            if self.args.dryrun:
                break

        return poison_delta, target_losses



    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle):
        """Take a step toward minmizing the current target loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        # Add adversarial pattern
        poison_slices, batch_positions = [], []
        for batch_id, image_id in enumerate(ids.tolist()):
            lookup = kettle.poison_lookup.get(image_id)
            if lookup is not None:
                poison_slices.append(lookup)
                batch_positions.append(batch_id)

        # This is a no-op in single network brewing
        # In distributed brewing, this is a synchronization operation
        inputs, labels, poison_slices, batch_positions, randgen = victim.distributed_control(
            inputs, labels, poison_slices, batch_positions)

        if len(batch_positions) > 0:
            if self.num_groups == 1:
                # Original behavior: process entire batch together
                return self._process_single_group_batch(poison_delta, poison_bounds, inputs, labels, 
                                                      poison_slices, batch_positions, victim, kettle, randgen)
            else:
                # Focused attack: split batch by groups and process separately
                return self._process_mixed_batch_with_splitting(poison_delta, poison_bounds, inputs, labels,
                                                              poison_slices, batch_positions, victim, kettle, randgen)
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)
            return loss.item(), prediction.item()

    def _process_single_group_batch(self, poison_delta, poison_bounds, inputs, labels, 
                                   poison_slices, batch_positions, victim, kettle, randgen):
        """Process a batch with original logic (backward compatible)."""
        delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
        if self.args.clean_grad:
            delta_slice = torch.zeros_like(delta_slice)
        delta_slice.requires_grad_()
        poison_images = inputs[batch_positions]
        inputs[batch_positions] += delta_slice

        # Perform differentiable data augmentation
        if self.args.paugment:
            inputs = kettle.augment(inputs, randgen=randgen)

        # Define the loss objective and compute gradients
        batch_target_grad, batch_target_clean_grad, batch_target_gnorm = self._get_target_grad_for_batch(poison_slices)
        
        closure = self._define_objective(inputs, labels, self.targets, self.intended_classes,
                                         self.true_classes)
        loss, prediction = victim.compute(closure, batch_target_grad, batch_target_clean_grad,
                                          batch_target_gnorm)
        delta_slice = victim.sync_gradients(delta_slice)

        if self.args.clean_grad:
            delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

        # Update Step
        if self.args.attackoptim in ['PGD', 'GD']:
            delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)
            poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
        elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            _, c, h, w = delta_slice.data.shape
            reg = torch.zeros((c, h, w)).to(device=self.setup['device'])
            for i in range(h):
                reg[:, i, :] = self.args.reg_factor * (h - i - 1)
            reg = reg.view(1, c, h, w)
            delta_slice.grad += 2 * delta_slice.data * reg
            poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
            poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
        else:
            raise NotImplementedError('Unknown attack optimizer.')

        return loss.item(), prediction.item()

    def _process_mixed_batch_with_splitting(self, poison_delta, poison_bounds, inputs, labels,
                                          poison_slices, batch_positions, victim, kettle, randgen):
        """Process a mixed batch by splitting into groups for focused attacks."""
        # Check if we can use the fast sequential path
        if self.args.allocation_strategy == 'sequential' and len(poison_slices) > 1:
            # Try fast path for sequential allocation
            fast_result = self._try_fast_sequential_processing(
                poison_delta, poison_bounds, inputs, labels,
                poison_slices, batch_positions, victim, kettle, randgen
            )
            if fast_result is not None:
                return fast_result

        # General path: Group poisons by their target group using efficient data structures
        group_poison_slices = {}
        group_batch_positions = {}
        group_sizes = {}
        
        for i, poison_id in enumerate(poison_slices):
            group_id = self.poison_to_group[poison_id]
            if group_id not in group_poison_slices:
                group_poison_slices[group_id] = []
                group_batch_positions[group_id] = []
                group_sizes[group_id] = 0
            
            group_poison_slices[group_id].append(poison_id)
            group_batch_positions[group_id].append(batch_positions[i])
            group_sizes[group_id] += 1

        total_loss = 0.0
        total_prediction = 0
        num_processed = 0

        # Process each group separately
        for group_id in group_poison_slices.keys():
            # Process this group
            group_loss, group_prediction = self._process_group_subset(
                poison_delta, poison_bounds, inputs, labels,
                group_poison_slices[group_id], group_batch_positions[group_id], group_id,
                victim, kettle, randgen
            )

            group_size = group_sizes[group_id]
            total_loss += group_loss * group_size  # Weight by group size
            total_prediction += group_prediction
            num_processed += group_size

        # Return weighted average loss and total predictions
        avg_loss = total_loss / num_processed if num_processed > 0 else 0.0
        return avg_loss, total_prediction

    def _try_fast_sequential_processing(self, poison_delta, poison_bounds, inputs, labels,
                                      poison_slices, batch_positions, victim, kettle, randgen):
        """Fast path for sequential allocation: use index ranges instead of iteration."""
        # For sequential allocation, poison IDs are consecutive within each group
        # So we can identify group boundaries without checking every poison
        
        min_poison_id = min(poison_slices)
        max_poison_id = max(poison_slices)
        
        # Check if this batch spans multiple groups
        min_group = self.poison_to_group[min_poison_id]
        max_group = self.poison_to_group[max_poison_id]
        
        if min_group == max_group:
            # Homogeneous batch - no splitting needed, but still use group-specific gradients
            loss, prediction = self._process_group_subset(
                poison_delta, poison_bounds, inputs, labels,
                poison_slices, batch_positions, min_group, victim, kettle, randgen
            )
            return loss, prediction
        
        # Mixed batch with sequential allocation - find group boundaries
        # Since allocation is sequential, we can find boundaries efficiently
        group_ranges = self._find_sequential_group_ranges(poison_slices)
        
        if group_ranges is None:
            # Batch is not contiguous in poison space - fall back to general path
            return None
            
        # Process each contiguous group range
        total_loss = 0.0
        total_prediction = 0
        num_processed = 0
        
        for group_id, (start_idx, end_idx) in group_ranges.items():
            # Extract slice for this group
            group_poison_slices = poison_slices[start_idx:end_idx]
            group_batch_positions = batch_positions[start_idx:end_idx]
            
            # Process this group
            group_loss, group_prediction = self._process_group_subset(
                poison_delta, poison_bounds, inputs, labels,
                group_poison_slices, group_batch_positions, group_id,
                victim, kettle, randgen
            )
            
            group_size = end_idx - start_idx
            total_loss += group_loss * group_size
            total_prediction += group_prediction
            num_processed += group_size
        
        # Return weighted average loss and total predictions
        avg_loss = total_loss / num_processed if num_processed > 0 else 0.0
        return avg_loss, total_prediction

    def _find_sequential_group_ranges(self, poison_slices):
        """Find contiguous group ranges for sequential allocation."""
        # Check if poison_slices are in sequential order
        if poison_slices != sorted(poison_slices):
            return None  # Not sequential in this batch
            
        # Find group boundaries
        group_ranges = {}
        current_group = self.poison_to_group[poison_slices[0]]
        start_idx = 0
        
        for i, poison_id in enumerate(poison_slices):
            poison_group = self.poison_to_group[poison_id]
            
            if poison_group != current_group:
                # Group boundary found
                group_ranges[current_group] = (start_idx, i)
                current_group = poison_group
                start_idx = i
        
        # Add the last group
        group_ranges[current_group] = (start_idx, len(poison_slices))
        
        return group_ranges

    def _process_group_subset(self, poison_delta, poison_bounds, inputs, labels,
                            group_poison_slices, group_batch_positions, group_id,
                            victim, kettle, randgen):
        """Process a subset of the batch belonging to a specific group."""
        # Get group-specific target gradients
        group_info = self.target_grad_groups[group_id]
        target_grad = group_info['grad']
        target_gnorm = group_info['gnorm']
        target_clean_grad = self.target_clean_grad_groups.get(group_id) if self.args.repel != 0 else None

        # Extract delta slice for this group
        delta_slice = poison_delta[group_poison_slices].detach().to(**self.setup)
        if self.args.clean_grad:
            delta_slice = torch.zeros_like(delta_slice)
        delta_slice.requires_grad_()

        # Extract only the relevant inputs and labels for this group
        poison_images = inputs[group_batch_positions]
        group_inputs = poison_images + delta_slice
        group_labels = labels[group_batch_positions]

        # Perform differentiable data augmentation
        if self.args.paugment:
            group_inputs = kettle.augment(group_inputs, randgen=randgen)

        # Define the loss objective and compute gradients
        closure = self._define_objective(group_inputs, group_labels, self.targets, self.intended_classes,
                                         self.true_classes)
        loss, prediction = victim.compute(closure, target_grad, target_clean_grad, target_gnorm)
        delta_slice = victim.sync_gradients(delta_slice)

        if self.args.clean_grad:
            delta_slice.data = poison_delta[group_poison_slices].detach().to(**self.setup)

        # Update Step
        if self.args.attackoptim in ['PGD', 'GD']:
            delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)
            poison_delta[group_poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
        elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            _, c, h, w = delta_slice.data.shape
            reg = torch.zeros((c, h, w)).to(device=self.setup['device'])
            for i in range(h):
                reg[:, i, :] = self.args.reg_factor * (h - i - 1)
            reg = reg.view(1, c, h, w)
            delta_slice.grad += 2 * delta_slice.data * reg
            poison_delta.grad[group_poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
            poison_bounds[group_poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
        else:
            raise NotImplementedError('Unknown attack optimizer.')

        return loss.item(), prediction.item()

    def _define_objective():
        """Implement the closure here."""
        def closure(model, criterion, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            raise NotImplementedError()
            return target_loss.item(), prediction.item()

    def _pgd_step(self, delta_slice, poison_imgs, tau, dm, ds):
        """PGD step."""
        with torch.no_grad():
            # Gradient Step
            if self.args.attackoptim == 'GD':
                delta_slice.data -= delta_slice.grad * tau
            else:
                delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step
            delta_slice.data = torch.max(torch.min(delta_slice, self.args.eps /
                                                   ds / 255), -self.args.eps / ds / 255)
            delta_slice.data = torch.max(torch.min(delta_slice, (1 - dm) / ds -
                                                   poison_imgs), -dm / ds - poison_imgs)
        return delta_slice
