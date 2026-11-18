"""Experimental constant-shape BF16 projection, independent of outer batch.

Pads each sequence into fixed row tiles then batches a fixed number of tiles
per bmm. Tile placement can still affect a device kernel: validate slot and
neighbor invariance, full-model probabilities and gradients before admission.
Not installed in any production module or inference worker.
"""
import torch
from torch.nn import functional as F


def fixed_shape_linear(inputs, weight, bias=None, *, tile_rows=128, tile_batch=8):
    if (inputs.ndim != 3 or weight.ndim != 2 or inputs.shape[-1] != weight.shape[-1]
            or min(inputs.shape) < 1 or weight.shape[0] < 1
            or inputs.dtype != weight.dtype or inputs.device != weight.device
            or type(tile_rows) is not int or tile_rows < 1
            or type(tile_batch) is not int or tile_batch < 1):
        raise ValueError('aligned nonempty batch/sequence/input and weight with positive fixed tiles required')
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != weight.dtype
                             or bias.device != weight.device):
        raise ValueError('aligned projection bias required')
    batch, length, width = inputs.shape
    padded_length = ((length + tile_rows - 1) // tile_rows) * tile_rows
    padded = F.pad(inputs, (0, 0, 0, padded_length - length))
    tiles = padded.reshape(-1, tile_rows, width)
    count = tiles.shape[0]
    tiles = F.pad(tiles, (0, 0, 0, 0, 0, (-count) % tile_batch))
    outputs = []
    weights = weight.t()[None].expand(tile_batch, -1, -1)
    for offset in range(0, tiles.shape[0], tile_batch):
        outputs.append(torch.bmm(tiles[offset:offset+tile_batch], weights))
    output = torch.cat(outputs, 0)[:count].reshape(batch, padded_length, weight.shape[0])[:, :length]
    return output if bias is None else output + bias
