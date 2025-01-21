from typing import List, Tuple, Union

import torch


def ccr(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute first principal component and subtract it from matrix.

    Args:
        x: input matrix
    Returns:
        Subtracted matrix and first principal component
    """
    fp = x.svd()[2][:, 0]
    x_red = x - x.mm(fp.unsqueeze(1)).squeeze(1).ger(fp)

    return x_red, fp


def _pad_tensors(tensors: List[torch.Tensor],
                 dim: int,
                 padding_value: Union[int, float] = 0) -> List[torch.Tensor]:
    result: List[torch.Tensor] = []
    max_size = max(t.size(dim) for t in tensors)
    for tensor in tensors:
        transposed = tensor.transpose(0, dim)
        out_dims = (max_size, ) + transposed.size()[1:]
        out = transposed.data.new(*out_dims).fill_(  #type: ignore
            padding_value)
        out[:transposed.size(0)] = transposed

        result.append(out.transpose(0, dim))

    return result


def pad_tensors(tensors: List[torch.Tensor],
                dims: List[int],
                padding_value: Union[int, float] = 0) -> List[torch.Tensor]:
    """Pad tensors to all be of the same size."""
    for dim in dims:
        tensors = _pad_tensors(tensors, dim, padding_value)

    return tensors
