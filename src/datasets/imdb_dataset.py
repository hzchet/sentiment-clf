import random
import json
import logging
import os

import torch

from src.base.base_dataset import BaseDataset
from src.base.base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class IMDbDataset(BaseDataset):
    def __init__(self, tokenizer: BaseTokenizer, split: str, *args, **kwargs):
        assert split in ('train', 'test'), "Invalid 'split' value. Please use 'train' or 'test'."
        self.tokenizer = tokenizer
        index = self._create_or_load_index(split)
        
        super().__init__(index, *args, **kwargs)
        
    def _create_or_load_index(self, split: str):
        path_to_index = f'saved/IMDb/index_{split}.json'
        if os.path.exists(path_to_index):
            logger.info(f'Loading index from "saved/IMDb/{split}.json"...')
            with open(path_to_index, 'r') as f:
                index = json.load(f)
        else:
            logger.info(f'Creating index at "saved/IMDb/{split}.json"...')
            path_to_data = f'data/IMDb/{split}'
            index = []
            with open(f'{path_to_data}/positive', 'r') as f:
                for text in f:
                    data_dict = self.tokenizer.encode(text, return_tensors=False)
                    data_dict.update({"text": text.strip(), "label": 1})
                    index.append(data_dict)
            
            with open(f'{path_to_data}/negative', 'r') as f:
                for text in f:
                    data_dict = self.tokenizer.encode(text, return_tensors=False)
                    data_dict.update({"text": text.strip(), "label": 0})
                    index.append(data_dict)
            
            random.shuffle(index)
            os.makedirs(f'saved/IMDb', exist_ok=True)
            with open(path_to_index, 'w') as f:
                json.dump(index, f, indent=2)

        for ind in range(len(index)):
            index[ind]["input_ids"] = torch.tensor(index[ind]["input_ids"])
            index[ind]["attention_mask"] = torch.tensor(index[ind]["attention_mask"])
        
        return index
