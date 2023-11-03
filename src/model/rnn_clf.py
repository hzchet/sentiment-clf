from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.base.base_model import BaseModel
from src.model.vanilla_rnn import MyVanillaRNN
from src.model.lstm import MyLSTM
from src.model.multilayer_lstm import MyMultilayerLSTM
from src.model.bi_lstm import MyBidirectionalLSTM


_RNN_TYPE_MATCHING = {
    'vanilla_rnn': MyVanillaRNN,
    'lstm': MyLSTM,
    'multilayer_lstm': MyMultilayerLSTM,
    'bi_lstm': MyBidirectionalLSTM
}


class RNNClassifier(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        n_class: int = 2,
        dropout_p: float = 0.0,
        padding_idx: int = 0,
        rnn_kwargs: Dict = None,
        **batch
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx, n_class, **batch)
        
        assert rnn_type in _RNN_TYPE_MATCHING
        if rnn_kwargs is None:
            rnn_kwargs = dict()
        self.rnn = _RNN_TYPE_MATCHING[rnn_type](input_size, hidden_size, **rnn_kwargs)
        
        if rnn_type == 'bi_lstm' and 'bidirectional' in rnn_kwargs and rnn_kwargs['bidirectional']:
            self.head = nn.Sequential(nn.Linear(hidden_size * 2, n_class), nn.Dropout(dropout_p))
        else:
            self.head = nn.Sequential(nn.Linear(hidden_size, n_class), nn.Dropout(dropout_p))
        
    def forward(self, input_ids, attention_mask, **batch):
        assert input_ids.shape[0] == attention_mask.shape[0], 'wtf yo'
        B = input_ids.shape[0]
        x = self.embedding(input_ids)
        output, h = self.rnn(x)
        
        output = torch.stack(
            [output[i, int(attention_mask[i].sum()) - 1, :] for i in range(B)], dim=0
        )
        
        return {"logits": self.head(output)}
