"""Repeatable code parts concerning optimization and training schedules."""


import torch

import datetime
from .utils import print_and_save_stats, pgd_step

from ..consts import NON_BLOCKING, BENCHMARK, DEBUG_TRAINING
torch.backends.cudnn.benchmark = BENCHMARK


def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-basic':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        #optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)
        all_parameters = list(model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = torch.optim.AdamW(params, lr=defs.lr, weight_decay=defs.weight_decay)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )

    if defs.scheduler == 'cyclic':
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        print(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)

    elif defs.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, defs.epochs, eta_min=defs.lr / 100)

        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler


def create_extended_scheduler(optimizer, defs, original_epochs, max_additional_epochs):
    """Create a scheduler for extended training with longer duration and lower minimum LR."""
    total_epochs = original_epochs + max_additional_epochs
    
    if defs.scheduler == 'linear':
        # Add an additional milestone at the end for further decay
        original_milestones = [original_epochs // 2.667, original_epochs // 1.6, original_epochs // 1.142]
        # Add a final milestone at 90% of total extended epochs
        extended_milestones = [int(0.2 * total_epochs), int(0.4 * total_epochs), int(0.65 * total_epochs), int(0.85 * total_epochs)]
        extended_milestones = [int(m) for m in original_milestones] + extended_milestones
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=extended_milestones, gamma=0.1)
        print(f'Extended linear scheduler: milestones at {extended_milestones} for {total_epochs} total epochs')
        
    elif defs.scheduler == 'cosine':
        # Use 10x lower minimum LR for extended training
        extended_eta_min = defs.lr / 1000  # 10x lower than original eta_min (lr/100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, eta_min=extended_eta_min)
        print(f'Extended cosine scheduler: {total_epochs} epochs with eta_min={extended_eta_min:.8f} (10x lower)')
        
    else:
        # For other schedulers, use the original logic
        if defs.scheduler == 'cyclic':
            effective_batches = (50_000 // defs.batch_size) * total_epochs
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                          step_size_up=effective_batches // 2,
                                                          cycle_momentum=True if defs.optimizer in ['SGD'] else False)
        elif defs.scheduler == 'none':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[10_000, 15_000, 25_000], gamma=1)
    
    return scheduler


def run_step(kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, ablation=True, max_epoch=None):

    epoch_loss, total_preds, correct_preds = 0, 0, 0
    if DEBUG_TRAINING:
        data_timer_start = torch.cuda.Event(enable_timing=True)
        data_timer_end = torch.cuda.Event(enable_timing=True)
        forward_timer_start = torch.cuda.Event(enable_timing=True)
        forward_timer_end = torch.cuda.Event(enable_timing=True)
        backward_timer_start = torch.cuda.Event(enable_timing=True)
        backward_timer_end = torch.cuda.Event(enable_timing=True)

        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0

        data_timer_start.record()

    if kettle.args.ablation < 1.0:
        # run ablation on a subset of the training set
        loader = kettle.partialloader
    else:
        loader = kettle.trainloader

    for batch, (inputs, labels, ids) in enumerate(loader):
        # Prep Mini-Batch
        model.train()
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

        if DEBUG_TRAINING:
            data_timer_end.record()
            forward_timer_start.record()

        # Add adversarial pattern
        if poison_delta is not None:
            poison_slices, batch_positions = [], []
            for batch_id, image_id in enumerate(ids.tolist()):
                lookup = kettle.poison_lookup.get(image_id)
                if lookup is not None:
                    poison_slices.append(lookup)
                    batch_positions.append(batch_id)
            # Python 3.8:
            # twins = [(b, l) for b, i in enumerate(ids.tolist()) if l:= kettle.poison_lookup.get(i)]
            # poison_slices, batch_positions = zip(*twins)

            if batch_positions:
                inputs[batch_positions] += poison_delta[poison_slices].to(**kettle.setup)

        # Add data augmentation
        if defs.augmentations:  # defs.augmentations is actually a string, but it is False if --noaugment
            inputs = kettle.augment(inputs)

        # Does adversarial training help against poisoning?
        for _ in range(defs.adversarial_steps):
            inputs = pgd_step(inputs, labels, model, loss_fn, kettle.dm, kettle.ds,
                              eps=kettle.args.eps, tau=kettle.args.tau)

        # Get loss
        outputs = model(inputs)
        loss = loss_fn(model, outputs, labels)
        if DEBUG_TRAINING:
            forward_timer_end.record()
            backward_timer_start.record()

        loss.backward()

        # Enforce batch-wise privacy if necessary
        # This is a defense discussed in Hong et al., 2020
        # We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
        # This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
        # of noise to the gradient signal
        with torch.no_grad():
            if defs.privacy['clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), defs.privacy['clip'])
            if defs.privacy['noise'] is not None:
                # generator = torch.distributions.laplace.Laplace(torch.as_tensor(0.0).to(**kettle.setup),
                #                                                 kettle.defs.privacy['noise'])
                for param in model.parameters():
                    # param.grad += generator.sample(param.shape)
                    noise_sample = torch.randn_like(param) * defs.privacy['clip'] * defs.privacy['noise']
                    param.grad += noise_sample


        optimizer.step()

        predictions = torch.argmax(outputs.data, dim=1)
        total_preds += labels.size(0)
        correct_preds += (predictions == labels).sum().item()
        epoch_loss += loss.item()

        if DEBUG_TRAINING:
            backward_timer_end.record()
            torch.cuda.synchronize()
            stats['data_time'] += data_timer_start.elapsed_time(data_timer_end)
            stats['forward_time'] += forward_timer_start.elapsed_time(forward_timer_end)
            stats['backward_time'] += backward_timer_start.elapsed_time(backward_timer_end)

            data_timer_start.record()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if kettle.args.dryrun:
            break
    
    # Step scheduler normally - extended schedulers will handle the extended training period
    if defs.scheduler == 'linear' or defs.scheduler == 'cosine':
        scheduler.step()

    # Fix: Use the actual max_epoch being trained, not defs.epochs which might be different during retraining
    # We need to pass max_epoch to this function, but for now, let's just validate periodically
    if epoch % defs.validate == 0:
        valid_acc, valid_loss = run_validation(model, criterion, kettle.validloader, kettle.setup, kettle.args.dryrun)
        target_acc, target_loss, target_clean_acc, target_clean_loss = check_targets(
            model, criterion, kettle.targetset, kettle.poison_setup['intended_class'],
            kettle.poison_setup['target_class'],
            kettle.setup)
    else:
        valid_acc, valid_loss = None, None
        target_acc, target_loss, target_clean_acc, target_clean_loss = [None] * 4

    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                         valid_acc, valid_loss,
                         target_acc, target_loss, target_clean_acc, target_clean_loss)

    if DEBUG_TRAINING:
        print(f"Data processing: {datetime.timedelta(milliseconds=stats['data_time'])}, "
              f"Forward pass: {datetime.timedelta(milliseconds=stats['forward_time'])}, "
              f"Backward Pass and Gradient Step: {datetime.timedelta(milliseconds=stats['backward_time'])}")
        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0


def run_validation(model, criterion, dataloader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, targets).item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            if dryrun:
                break

    accuracy = correct / total
    loss_avg = loss / (i + 1)
    return accuracy, loss_avg

def check_targets(model, criterion, targetset, intended_class, original_class, setup, batch_size=10):
    """Get accuracy and loss for all targets on their intended class, processed in batches."""
    model.eval()
    if len(targetset) > 0:

        target_images = torch.stack([data[0] for data in targetset]).to(**setup)
        intended_labels = torch.tensor(intended_class).to(device=setup['device'], dtype=torch.long)
        original_labels = torch.stack([torch.as_tensor(data[1], device=setup['device'], dtype=torch.long) for data in targetset])
        
        # Initialize accumulators
        total_correct_intended = 0
        total_correct_clean = 0
        total_loss_intended = 0.0
        total_loss_clean = 0.0
        total_samples = len(targetset)
        
        with torch.no_grad():
            # Process targets in batches
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_size_actual = end_idx - start_idx
                
                # Slice the tensors for this batch
                batch_images = target_images[start_idx:end_idx]
                batch_intended_labels = intended_labels[start_idx:end_idx]
                batch_original_labels = original_labels[start_idx:end_idx]
                
                # Forward pass
                outputs = model(batch_images)
                predictions = torch.argmax(outputs, dim=1)
                
                # Compute metrics for this batch
                loss_intended_batch = criterion(outputs, batch_intended_labels)
                correct_intended_batch = (predictions == batch_intended_labels).sum().item()
                
                loss_clean_batch = criterion(outputs, batch_original_labels)
                correct_clean_batch = (predictions == batch_original_labels).sum().item()
                
                # Accumulate results
                total_correct_intended += correct_intended_batch
                total_correct_clean += correct_clean_batch
                total_loss_intended += loss_intended_batch.item() * batch_size_actual
                total_loss_clean += loss_clean_batch.item() * batch_size_actual
        
        # Compute final averages
        accuracy_intended = total_correct_intended / total_samples
        accuracy_clean = total_correct_clean / total_samples
        loss_intended = total_loss_intended / total_samples
        loss_clean = total_loss_clean / total_samples
        
        # print(f'Raw softmax output is {torch.softmax(outputs, dim=1)}, intended: {intended_class}')
        
        return accuracy_intended, loss_intended, accuracy_clean, loss_clean
    else:
        return 0, 0, 0, 0