"""Linear Recurrent Unit (LRU) implementation in PyTorch.

Based on the paper "Resurrecting Recurrent Neural Networks for Long Sequences" by Orvieto et al. (2023).
https://arxiv.org/pdf/2303.06349

This implementation closely follows the JAX implementation from:
https://github.com/NicolasZucchet/minimal-LRU
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LRULayer(nn.Module):
    """Core LRU computation layer.
    
    This implements the core recurrent computation of the LRU.
    """
    
    def __init__(self, d_model, d_state=64, r_min=0.0, r_max=1.0, max_phase=6.28):
        """Initialize LRU computation layer.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            r_min: Minimum radius for initialization
            r_max: Maximum radius for initialization
            max_phase: Maximum phase for initialization
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Initialize Lambda (diagonal state matrix) - complex valued
        u1 = torch.rand(d_state)
        u2 = torch.rand(d_state)
        
        # Sample nu (controls radius/decay)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        
        # Sample theta (controls phase/frequency)
        theta_log = torch.log(u2 * max_phase)
        
        # Store as learnable parameters
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)
        
        # Initialize input matrix B - complex valued
        B_re = torch.randn(d_model, d_state) / math.sqrt(2 * d_state)
        B_im = torch.randn(d_model, d_state) / math.sqrt(2 * d_state)
        self.B_re = nn.Parameter(B_re)
        self.B_im = nn.Parameter(B_im)
        
        # Initialize output matrix C - complex valued
        C_re = torch.randn(d_model, d_state) / math.sqrt(d_state)
        C_im = torch.randn(d_model, d_state) / math.sqrt(d_state)
        self.C_re = nn.Parameter(C_re)
        self.C_im = nn.Parameter(C_im)
        
        # Skip connection parameter
        self.D = nn.Parameter(torch.randn(d_model))
    
    def get_lambda(self):
        """Get the diagonal state matrix Lambda."""
        return torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))
    
    def get_B(self):
        """Get complex input matrix B."""
        return self.B_re + 1j * self.B_im
    
    def get_C(self):
        """Get complex output matrix C."""
        return self.C_re + 1j * self.C_im
    
    def forward(self, x):
        """Forward pass through LRU layer.
        
        Args:
            x: Input tensor of shape (B, L, H)
            
        Returns:
            y: Output tensor of shape (B, L, H)
        """
        B, L, H = x.shape
        
        # Get parameters
        Lambda = self.get_lambda()  # (N,)
        B_matrix = self.get_B()     # (H, N)
        C_matrix = self.get_C()     # (H, N)
        
        # Vectorized parallel computation using FFT convolution
        # Compute impulse response (convolution kernel) for all dimensions at once
        k = torch.arange(L, device=x.device, dtype=torch.float32)
        Lambda_k = Lambda.unsqueeze(0) ** k.unsqueeze(-1)  # (L, N)
        
        # Compute kernels for all output dimensions: (H, L)
        # kernel[h, k] = sum_n C[h,n] * B[h,n] * Lambda[n]^k
        kernels = torch.einsum('hn,ln->hl', C_matrix * B_matrix, Lambda_k).real  # (H, L)
        
        # Vectorized FFT convolution across all dimensions
        x_f = torch.fft.rfft(x, n=2*L, dim=1)  # (B, L_freq, H)
        kernels_f = torch.fft.rfft(kernels, n=2*L, dim=1)  # (H, L_freq)
        
        # Broadcasting: (B, L_freq, H) * (H, L_freq) -> (B, L_freq, H)
        y_f = x_f * kernels_f.unsqueeze(0).transpose(1, 2)  # (B, L_freq, H)
        y = torch.fft.irfft(y_f, n=2*L, dim=1)[:, :L, :]  # (B, L, H)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)  # (B, L, H)
        
        return y


class LRU(nn.Module):
    """Linear Recurrent Unit (LRU) basic block.
    
    This is the core LRU computation, similar to S4D - just the state space model.
    GLU, normalization, and dropout are handled at the model level.
    """
    
    def __init__(self, d_model, d_state=64, dropout=0.1, r_min=0.0, r_max=1.0, max_phase=6.28):
        """Initialize LRU basic block.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            dropout: Dropout rate (for compatibility, but handled at model level)
            r_min: Minimum radius for initialization
            r_max: Maximum radius for initialization
            max_phase: Maximum phase for initialization
        """
        super().__init__()
        
        # Core LRU computation (just like S4D is just the core SSM)
        self.lru_layer = LRULayer(d_model, d_state, r_min, r_max, max_phase)
        
        # Optional: simple output linear layer (like S4D has output_linear)
        self.output_linear = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GLU(dim=-1),
        )
    
    def forward(self, x):
        """Forward pass through LRU basic block.
        
        Args:
            x: Input tensor of shape (B, L, H)
            
        Returns:
            y: Output tensor of shape (B, L, H)
            state: Dummy state (None) for compatibility with S4D interface
        """
        # Apply core LRU computation
        y = self.lru_layer(x)  # (B, L, H)
        
        # Apply output transformation (like S4D)
        y = self.output_linear(y)  # (B, L, H)
        
        return y, None  # Return dummy state for compatibility