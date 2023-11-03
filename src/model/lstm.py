from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MyLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        disabled_gate: str = None
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        assert disabled_gate in ('input', 'forget', 'cell', 'output', None)
        self.disabled_gate = disabled_gate
        self.input_gate = nn.Linear(input_size, hidden_size)
        self.h_input_gate = nn.Linear(hidden_size, hidden_size)
        
        self.forget_gate = nn.Linear(input_size, hidden_size)
        self.h_forget_gate = nn.Linear(hidden_size, hidden_size)
        
        self.cell_gate = nn.Linear(input_size, hidden_size)
        self.h_cell_gate = nn.Linear(hidden_size, hidden_size)
        
        self.output_gate = nn.Linear(input_size, hidden_size)
        self.h_output_gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, hidden_tuple: Optional[Tuple[Tensor, Tensor]] = None):
        if hidden_tuple is None:
            h_0, c_0 = None, None
        else:
            h_0, c_0 = hidden_tuple

        assert len(x.shape) == 3 and x.shape[-1] == self.input_size, f'x.shape={x.shape}, input_size={self.input_size}'
        
        if h_0 is None:
            h_0 = torch.zeros(x.shape[0], self.hidden_size).to(x.device)
        if c_0 is None:
            c_0 = torch.zeros(x.shape[0], self.hidden_size).to(x.device)
            
        assert h_0.shape[-1] == c_0.shape[-1] == self.hidden_size

        h = h_0
        c = c_0
        output = []
        for t in range(int(x.shape[1])):
            i, f, g, o = 1, 1, 1, 1
            if self.disabled_gate != 'input':
                i = F.sigmoid(self.input_gate(x[:, t, :]) + self.h_input_gate(h))
            if self.disabled_gate != 'forget':
                f = F.sigmoid(self.forget_gate(x[:, t, :]) + self.h_forget_gate(h))
            if self.disabled_gate != 'cell':
                g = F.tanh(self.cell_gate(x[:, t, :]) + self.h_cell_gate(h))
            if self.disabled_gate != 'output':
                o = F.sigmoid(self.output_gate(x[:, t, :]) + self.h_output_gate(h))
            
            c = f * c + i * g
            h = o * F.tanh(c)
            
            output.append(h.unsqueeze(1))
        
        output = torch.cat(output, dim=1)
        return output, (h, c)
