import os
import re
from typing import List, Union, Dict

import numpy as np
from torch import Tensor


class BaseTokenizer:
    def __init__(self, path_to_data: Union[str, List[str]], vocab_size: int = 100):
        assert 0 < vocab_size <= 30_000, "The 'vocab_size' must be a positive integer no greater than 30,000."
        
        self.path_to_data = self._assert_path(path_to_data)
        self.vocab_size = vocab_size
    
    def encode(self, text: Union[str, List[str]]) -> Dict[str, Tensor]:
        raise NotImplementedError()
    
    def decode(self, token_ids: Union[int, List[int], np.ndarray, Tensor]) -> str:
        raise NotImplementedError()

    def _assert_path(self, path_to_data: Union[str, List[str]]):
        if isinstance(path_to_data, str):
            path_to_data = [path_to_data]
        
        for path in path_to_data:
            assert os.path.exists(path), "The 'path_to_data' must contain valid paths to the train data."
                
        return path_to_data
    
    def __len__(self) -> int:
        return self.vocab_size
    
    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        return text

    @staticmethod
    def pre_tokenize(text: str) -> List[str]:
        return text.split()
