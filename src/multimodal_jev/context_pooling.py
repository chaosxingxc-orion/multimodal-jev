"""Complete-prefill history readout shared with the candidate vLLM pooler.

The caller authenticates/loads the joint policy before constructing this object.
This module binds requests to that identity; it is not a weight loader.
"""
import torch
from torch import nn


class ContextHistoryReadout(nn.Module):
    def __init__(self, head, decision_token_id, policy_id, *, return_logits=False):
        super().__init__()
        if type(decision_token_id) is not int or decision_token_id <= 0:
            raise ValueError('positive registered marker ID required')
        if (type(policy_id) is not str or len(policy_id) != 64 or
                any(c not in '0123456789abcdef' for c in policy_id)):
            raise ValueError('authenticated joint policy digest required')
        self.head = head
        self.decision_token_id = decision_token_id
        self.policy_id = policy_id
        self.return_logits = return_logits

    def forward(self, hidden, lengths, prompts, extras):
        """Return one [decisions, 1] tensor per complete request.

        Flattened states must include every prompt token, including the audio
        expansion. Validate the complete batch before invoking the branch.
        """
        if (hidden.ndim != 2 or hidden.shape[1] != self.head.hidden_size or
                not hidden.is_floating_point() or not torch.isfinite(hidden).all()):
            raise ValueError('finite flattened backbone states required')
        if (not lengths or any(type(n) is not int or n <= 0 for n in lengths) or
                sum(lengths) != len(hidden) or len(prompts) != len(lengths) or
                len(extras) != len(lengths)):
            raise ValueError('complete aligned nonempty request lengths required')
        positions = []
        for n, prompt, extra in zip(lengths, prompts, extras, strict=True):
            if (prompt.ndim != 1 or prompt.dtype != torch.long or len(prompt) != n or
                    (prompt < 0).any()):
                raise ValueError('complete expanded integer prompt IDs required')
            if (type(extra) is not dict or
                    set(extra) != {'policy_id', 'marker_positions', 'expected_input_ids'} or
                    extra['policy_id'] != self.policy_id):
                raise ValueError('request must bind the loaded joint policy')
            expected = extra['expected_input_ids']
            if (type(expected) is not list or any(type(x) is not int for x in expected) or
                    prompt.tolist() != expected):
                raise ValueError('expanded prompt identity mismatch')
            pos = extra['marker_positions']
            actual = (prompt == self.decision_token_id).nonzero().flatten().tolist()
            if (type(pos) is not list or not pos or any(type(x) is not int for x in pos) or
                    pos != actual):
                raise ValueError('cover every marker once, in prompt order')
            positions.append(pos)
        outputs = []
        for chunk, pos in zip(hidden.split(lengths), positions, strict=True):
            indices = torch.tensor([pos], dtype=torch.long, device=chunk.device)
            valid = torch.ones((1, len(chunk)), dtype=torch.bool, device=chunk.device)
            values = self.head(chunk[None], indices, valid, return_logits=True) if self.return_logits else (
                self.head(chunk[None], indices, valid))
            if not torch.isfinite(values).all() or (
                    not self.return_logits and ((values < 0) | (values > 1)).any()):
                raise ValueError('invalid applicability readout')
            outputs.append(values[0, :, None])
        return outputs

