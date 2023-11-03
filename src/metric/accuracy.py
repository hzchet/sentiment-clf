import torch
from torch import Tensor

from src.base.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __call__(self, logits: Tensor, label: Tensor, **batch):
        return (logits.argmax(dim=1) == label).sum() / len(logits)
