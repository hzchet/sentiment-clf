import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MyVanillaRNN(nn.Module):
    def __init__(
        self,
        input_size: int, 
        hidden_size: int,
        nonlinearity: str = 'tanh'
    ):
        super().__init__()
        assert nonlinearity in ('tanh', 'relu')
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        self.hidden_fc = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc = nn.Linear(input_size, hidden_size, bias=True)
    
    def forward(self, x, h_0=None):
        assert len(x.shape) == 3 and x.shape[-1] == self.input_size
        
        if h_0 is None:
            h_0 = torch.zeros(x.shape[0], self.hidden_size).to(x.device)
        assert h_0.shape[-1] == self.hidden_size

        h = h_0
        output = []
        for t in range(int(x.shape[1])):
            h = self.fc(x[:, t, :]) + self.hidden_fc(h)
            output.append(h.unsqueeze(1))
            
            if self.nonlinearity == 'tanh':
                h = F.tanh(h)
            else:
                h = F.relu(h)
                
        return torch.cat(output, dim=1), h
