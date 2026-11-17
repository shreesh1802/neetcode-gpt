import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2.0/(fan_in + fan_out))
        W = torch.randn(fan_out,fan_in)*std
        return torch.round(W,decimals=4).tolist()
        pass

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2.0/fan_in)
        W = torch.randn(fan_out,fan_in)*std
        return torch.round(W,decimals=4).tolist()
        pass

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Use torch.manual_seed(0) once at the start.
        # Sample all weights in layer order with torch.randn(out_dim, in_dim) * std:
        # first (hidden_dim, input_dim), then (hidden_dim, hidden_dim).
        # Keep weights unrounded; do not call the init helpers, which reset the seed.
        # Then sample x = torch.randn(1, input_dim) without resetting the seed.
        # Apply x = torch.relu(x @ W.T) at every layer, with no biases.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # Use round(x.std().item(), 2) with PyTorch's default sample std.
        torch.manual_seed(0)
        weights = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            if init_type == "xavier":
                std = math.sqrt(2.0/(in_dim + out_dim))
            elif init_type == "kaiming":
                std = math.sqrt(2.0/in_dim)
            else:
                std = 1.0
            weights.append((torch.randn(out_dim,in_dim))*std)
        x = torch.randn(1,input_dim)

        stds= []
        for W in weights:
            x = torch.relu(x @ W.T)
            stds.append(round(x.std().item(),2))
        return stds
        pass
