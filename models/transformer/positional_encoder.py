import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, num_timesteps=100):
        super().__init__()
        self.d_model = d_model
        self.num_timesteps = num_timesteps

    def forward(self, x):
        pe = torch.zeros_like(x)
        for pos in range(0, self.num_timesteps):
            for i in range(0, self.d_model):
                if (i % 2 == 0):
                    pe[..., pos, i] = math.sin(
                        pos / 10000 ** (2 * i / self.d_model))
                else:
                    pe[..., pos, i] = math.cos(
                        pos / 10000 ** (2 * i / self.d_model))
        return x + pe
