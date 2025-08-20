import functools
import itertools
import operator
import os
from typing import Optional, Sequence, TypeAlias

import torch

MyPathLike: TypeAlias = str | os.PathLike[str]


def flatten_lists(lists_2d: list[list]) -> list:
    return functools.reduce(operator.iconcat, lists_2d, [])


def unique_consecutive[T](
    seq: Sequence[T], return_counts: bool = False
) -> tuple[T, ...] | tuple[tuple[T, ...], tuple[int, ...]]:
    unique, counts = zip(*[(el, len(list(gr))) for el, gr in itertools.groupby(seq)])
    if return_counts:
        return unique, counts
    return unique


def create_mask_from_lengths(
    lengths: torch.Tensor, max_length: Optional[int] = None
) -> torch.Tensor:
    """Create a mask from a tensor of lengths."""
    max_length = max_length or lengths.max().item()
    return torch.arange(max_length, device=lengths.device).expand(
        len(lengths), max_length
    ) < lengths.unsqueeze(1)
