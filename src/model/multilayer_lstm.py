from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.model.lstm import MyLSTM


class MyMultilayerLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1
    ):
        super().__init__()
        assert 1 <= num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_blocks = [MyLSTM(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            self.lstm_blocks.append(MyLSTM(hidden_size, hidden_size))
        
        self.lstm_blocks = nn.ModuleList(self.lstm_blocks)

    def forward(self, x, hidden_tuple: Optional[Tuple[Tensor, Tensor]] = None):
        output, h, c = None, [], []
        for block in self.lstm_blocks:
            output, (h_i, c_i) = block(x, hidden_tuple)
            x = output
            h.append(h_i.unsqueeze(1))
            c.append(c_i.unsqueeze(1))
            hidden_tuple = None
        
        h = torch.cat(h, dim=1)
        c = torch.cat(c, dim=1)
        return output, (h, c)
