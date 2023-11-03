from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.lstm import MyLSTM


class MyBidirectionalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.forward_lstm = MyLSTM(input_size, hidden_size)
        if bidirectional:
            self.backward_lstm = MyLSTM(input_size, hidden_size)
        
    def forward(self, x):
        output, (h, c) = self.forward_lstm(x)
        
        if self.bidirectional:
            output_2, (h2, c2) = self.backward_lstm(torch.flip(x, dims=(1, )))
            output = torch.cat([output, torch.flip(output_2, dims=(1,))], dim=2)
            h = torch.stack([h, h2], dim=1)
            c = torch.stack([c, c2], dim=1)
        
        return output, (h, c)
