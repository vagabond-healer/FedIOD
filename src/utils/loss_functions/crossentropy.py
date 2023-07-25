from torch import nn, Tensor
# import torch.nn.functional as F
# from torch.nn.functional import _reduction as _Reduction

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


# class _Loss(nn.Module):
#     reduction: str
#
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(_Loss, self).__init__()
#         if size_average is not None or reduce is not None:
#             self.reduction = _Reduction.legacy_get_string(size_average, reduce)
#         else:
#             self.reduction = reduction
#
#
# class _WeightedLoss(_Loss):
#     def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
#         self.register_buffer('weight', weight)
#
# class CrossEntropyLoss(_WeightedLoss):
#
#     __constants__ = ['ignore_index', 'reduction']
#     ignore_index: int
#
#     def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
#                  reduce=None, reduction: str = 'mean') -> None:
#         super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
#         self.ignore_index = ignore_index
#
#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         return F.cross_entropy(input, target, weight=self.weight,
#                                ignore_index=self.ignore_index, reduction=self.reduction)

