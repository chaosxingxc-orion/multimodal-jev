"""Opt-in complete-row readout, keeping the backbone batched.

This execution variant trims only trailing padding and preserves real positions.
It shares parameters and autograd; admission is separate from installation.
"""
import torch


def install_row_head(head):
    if getattr(head, '_jev_row_head', False):
        raise ValueError('row readout already installed')
    original = head.score_logits

    def score(hidden, positions, valid):
        head._validate(hidden, positions, valid)
        if not torch.equal(valid, valid.long().cumprod(-1).bool()):
            raise ValueError('row readout requires right padding')
        return torch.cat([original(hidden[i:i+1, :n], positions[i:i+1], valid[i:i+1, :n])
                          for i,n in enumerate(valid.sum(-1).tolist())])

    head.score_logits = score
    head._jev_row_head = True
    return dict(readout='per_request_complete_prefix',backbone='unchanged',parameters='shared')
