import random
import json
import re
import logging
import os

import torch

from src.base.base_dataset import BaseDataset
from src.base.base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class YelpDataset(BaseDataset):
    def __init__(self, tokenizer: BaseTokenizer, regexp: str = None, *args, **kwargs):
        self.tokenizer = tokenizer
        index = self._create_or_load_index(regexp)
        
        super().__init__(index, *args, **kwargs)
        
    def _create_or_load_index(self, regexp: str = None):
        path_to_index = f'saved/Yelp/index.json'
        if os.path.exists(path_to_index):
            logger.info(f'Loading index from "saved/Yelp/index.json"...')
            with open(path_to_index, 'r') as f:
                index = json.load(f)
        else:
            logger.info(f'Creating index at "saved/Yelp/index.json"...')
            path_to_data = f'data/Yelp/reviews.txt'
            index = []
            with open(f'{path_to_data}', 'r') as f:
                for text in f:
                    text_ = text
                    if regexp is not None:
                        text_ = re.sub(regexp, "", text)
                    data_dict = self.tokenizer.encode(text_, return_tensors=False)
                    data_dict.update({"text": text.strip(), "label": 1})
                    index.append(data_dict)
            
            os.makedirs(f'saved/Yelp', exist_ok=True)
            with open(path_to_index, 'w') as f:
                json.dump(index, f, indent=2)

        for ind in range(len(index)):
            index[ind]["input_ids"] = torch.tensor(index[ind]["input_ids"])
            index[ind]["attention_mask"] = torch.tensor(index[ind]["attention_mask"])
        
        return index
