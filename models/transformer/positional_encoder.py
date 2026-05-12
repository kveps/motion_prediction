import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, num_timesteps=100):
        super().__init__()
        # Precompute PE table once — shape [num_timesteps, d_model]
        pe = torch.zeros(num_timesteps, d_model)
        positions = torch.arange(num_timesteps).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [..., num_timesteps, d_model]
        return x + self.pe
