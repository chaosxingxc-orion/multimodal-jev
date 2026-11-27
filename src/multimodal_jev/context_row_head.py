"""Opt-in complete-row readout, keeping the backbone batched.

This execution variant trims only trailing padding and preserves real positions.
It shares parameters and autograd; admission is separate from installation.
"""
import torch


def install_row_head(head, *, single_validation=False):
    if getattr(head, '_jev_row_head', False):
        raise ValueError('row readout already installed')
    original = head.score_logits
    # The full tensor covers all trimmed rows, and the subsequent right-pad
    # check proves the shorter valid lengths still contain each marker.
    trimmed = head._score_validated if single_validation else original

    def score(hidden, positions, valid):
        head._validate(hidden, positions, valid)
        if not torch.equal(valid, valid.long().cumprod(-1).bool()):
            raise ValueError('row readout requires right padding')
        return torch.cat([trimmed(hidden[i:i+1, :n], positions[i:i+1], valid[i:i+1, :n])
                          for i,n in enumerate(valid.sum(-1).tolist())])

    head.score_logits = score
    head._jev_row_head = True
    return dict(readout='per_request_complete_prefix',backbone='unchanged',parameters='shared')
