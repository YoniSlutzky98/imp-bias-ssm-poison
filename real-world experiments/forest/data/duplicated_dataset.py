"""Custom dataset for handling duplicated poison samples."""

import torch
from torch.utils.data import Dataset


class DuplicatedPoisonDataset(Dataset):
    """Dataset that handles duplicated poison samples.
    
    This dataset presents poison samples twice:
    1. Once with their original ID (treated as clean)
    2. Once with a new ID (treated as poison candidate)
    
    The poison_lookup will only contain mappings for the new IDs.
    """
    
    def __init__(self, base_dataset, train_ids, original_poison_ids, poison_dirty_ids, id_offset):
        """Initialize the duplicated dataset.
        
        Args:
            base_dataset: The original dataset
            train_ids: All training IDs (clean + poison_clean + poison_dirty)
            original_poison_ids: Original poison sample IDs
            poison_dirty_ids: New IDs for poison copies (original_poison_ids + id_offset)
            id_offset: Offset used to create new IDs
        """
        self.base_dataset = base_dataset
        self.train_ids = train_ids
        self.original_poison_ids = original_poison_ids
        self.poison_dirty_ids = poison_dirty_ids
        self.id_offset = id_offset
        
        # Create mapping from train_ids index to actual data
        self.id_to_base_idx = {}
        
        for i, train_id in enumerate(train_ids):
            if train_id in poison_dirty_ids:
                # This is a "dirty" poison copy, map to original poison sample
                original_id = train_id - id_offset
                self.id_to_base_idx[i] = original_id
            else:
                # This is either clean or "clean" poison copy, use as-is
                self.id_to_base_idx[i] = train_id.item()
    
    def __len__(self):
        return len(self.train_ids)
    
    def __getitem__(self, idx):
        """Get item by index in the duplicated dataset."""
        train_id = self.train_ids[idx]
        base_idx = self.id_to_base_idx[idx]
        
        # Get the actual data from base dataset
        # Base dataset returns (data, label, original_idx) but we want to use our train_id
        data, label, _ = self.base_dataset[base_idx]
        
        # Return data, label, and the train_id (which will be used for poison lookup)
        return data, label, train_id
    
    def get_target(self, idx):
        """Get target information for compatibility."""
        train_id = self.train_ids[idx]
        base_idx = self.id_to_base_idx[idx]
        
        # Get label from base dataset using get_target method
        label, _ = self.base_dataset.get_target(base_idx)
        
        return label, train_id
    
    @property
    def classes(self):
        """Get classes from base dataset."""
        return self.base_dataset.classes
    
    @property
    def data_mean(self):
        """Get data mean from base dataset."""
        return self.base_dataset.data_mean
    
    @property
    def data_std(self):
        """Get data std from base dataset."""
        return self.base_dataset.data_std
