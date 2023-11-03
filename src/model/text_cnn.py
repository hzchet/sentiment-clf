from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.base_model import BaseModel


class TextCNN(BaseModel):
    def __init__(
        self, 
        num_embeddings: int,
        embedding_dim: int,
        channels: List[int],
        kernel_sizes: Union[List[int], List[Tuple[int, int]]],
        n_class: int = 2,
        padding_idx: int = 0,
        dropout_p: float = 0.5,
        **batch
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx, n_class, **batch)
        
        conv_blocks = []
        for in_channels, out_channels, kernel_size in zip(channels[:-1], channels[1:], kernel_sizes):
            conv_blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size),
                nn.ReLU()
            ))
        
        self.conv = nn.Sequential(*conv_blocks)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], n_class),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, input_ids, **batch):
        x = self.embedding(input_ids)
        x = self.conv(x.transpose(1, 2))
        x = F.max_pool1d(x, kernel_size=x.shape[-1])
        logits = self.head(x.squeeze(-1))
        
        return {"logits": logits}
