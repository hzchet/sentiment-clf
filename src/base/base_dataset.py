from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.base.base_tokenizer import BaseTokenizer


class BaseDataset(Dataset):
    def __init__(
        self,
        index: List[Dict],
        limit: int = None,
        *args,
        **kwargs
    ):
        super().__init__()
        
        if limit is not None:
            assert 0 <= limit <= len(index), "Invalid 'limit' value."
            index = index[:limit]
        
        self.index = index

    def __getitem__(self, ind: int):
        data_dict = self.index[ind]
        return data_dict

    def __len__(self) -> int:
        return len(self.index)
