from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: List[Dict], is_labeled: bool = True):
    """
    collate fields in dataset_items
    """
    
    texts = [item['text'] for item in dataset_items]
    input_ids = pad_sequence([item['input_ids'].squeeze(0) for item in dataset_items], batch_first=True)
    attention_masks = pad_sequence([item['attention_mask'].squeeze(0) for item in dataset_items], batch_first=True)
    
    data_dict = {
        'text': texts,
        'input_ids': input_ids,
        'attention_mask': attention_masks
    }
    
    if is_labeled:
        data_dict['label'] = torch.tensor([item['label'] for item in dataset_items])
        
    return data_dict
