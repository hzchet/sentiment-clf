from torch import Tensor
from torch.nn import CrossEntropyLoss


class CrossEntropyLossWrapper(CrossEntropyLoss):
    def forward(self, logits, label, **batch) -> Tensor:

        return super().forward(logits, label)
