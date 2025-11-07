"""LRU model implementation for vision tasks.

This file contains the LRU model architectures following the same pattern
as the S4 and Mamba2 models in this codebase.
"""

import torch
import torch.nn as nn

from .lru import LRU


class LRUModel(nn.Module):
    """LRU model for vision tasks.
    
    This model follows the same structure as S4Model, treating images as pixel sequences
    and processing them with LRU layers.
    """
    
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=128,
        n_layers=4,
        dropout=0.1,
        prenorm=False,
    ):
        """Initialize LRU model.
        
        Args:
            d_input: Input dimension (number of channels)
            d_output: Output dimension (number of classes)
            d_model: Hidden dimension for LRU layers
            n_layers: Number of LRU layers
            d_state: State dimension for LRU layers
            dropout: Dropout rate
            prenorm: Whether to use prenorm (True) or postnorm (False)
        """
        super().__init__()
        
        self.prenorm = prenorm
        
        # Linear encoder (d_input = channels for pixel sequence approach)
        self.encoder = nn.Linear(d_input, d_model)
        
        # Stack LRU layers as residual blocks (like S4Model)
        self.lru_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.lru_layers.append(
                LRU(d_model, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)
    
    def forward(self, x):
        """Forward pass through LRU model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, d_output)
        """
        B, d_input, h, w = x.shape
        x = x.view(B, h * w, d_input)  # (B, C, H, W) -> (B, L, d_input)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        # Process through LRU layers with residual connections (like S4Model)
        for layer, norm, dropout in zip(self.lru_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)
            
            # Apply LRU layer: we ignore the state input and output
            z, _ = layer(z)
            
            # Dropout on the output of the LRU layer
            z = dropout(z)
            
            # Residual connection
            x = z + x
            
            if not self.prenorm:
                # Postnorm
                x = norm(x)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

