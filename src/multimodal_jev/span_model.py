"""Single backbone invocation with simultaneous typed readouts; no decoding.

The final position can attend earlier latent span positions in the backbone.
Discrete answers are never fed back. Conditional categorical heads factorize;
this implementation alone is not evidence of CoT benefit.
"""
import torch
from torch import nn


class NativeAudioSpanModel(nn.Module):
    """Use the pretrained vocabulary head at causal predecessors of markers.

    Full vocabulary logits allow full-vocabulary masked SFT. Candidate-only
    normalization defines the finite decision policy, not a token decoder.
    """
    def __init__(self, conditional_model):
        super().__init__()
        self.backbone = conditional_model.model
        self.lm_head = conditional_model.lm_head

    def forward(self, inputs, positions, candidate_ids):
        if any(k in inputs for k in ('labels','past_key_values','use_cache','inputs_embeds','return_dict')):
            raise ValueError('label-free uncached inputs required')
        ids = inputs['input_ids']
        if (positions.dtype != torch.long or positions.ndim != 2 or
                positions.shape[0] != ids.shape[0] or positions.shape[1] == 0):
            raise ValueError('nonempty batched integer positions required')
        if ((positions < 0) | (positions >= ids.shape[-1])).any():
            raise ValueError('span outside sequence')
        if positions.shape[1] > 1 and (positions[:,1:] <= positions[:,:-1]).any():
            raise ValueError('ordered distinct span positions required')
        if not inputs['attention_mask'].gather(1,positions).bool().all():
            raise ValueError('span points into padding')
        if len(candidate_ids) != positions.shape[1]:
            raise ValueError('one candidate list per span required')
        for options in candidate_ids:
            if (len(options) < 2 or len(set(options)) != len(options) or
                    any(type(i) is not int or not 0 <= i < self.lm_head.out_features for i in options)):
                raise ValueError('distinct valid vocabulary candidates required')
        hidden = self.backbone(**inputs,use_cache=False,return_dict=True).last_hidden_state
        selected = hidden.gather(1,positions.unsqueeze(-1).expand(-1,-1,hidden.shape[-1]))
        vocabulary_logits = self.lm_head(selected).float()
        candidate_logprobs = tuple(torch.log_softmax(vocabulary_logits[:,j,options],dim=-1)
                                  for j,options in enumerate(candidate_ids))
        return vocabulary_logits, candidate_logprobs


class AudioSpanModel(nn.Module):
    def __init__(self, backbone, hidden_size, max_values):
        super().__init__()
        if max_values < 2:
            raise ValueError('at least two categorical values required')
        self.backbone = backbone  # Qwen3ASRModel, not its vocabulary generation head
        self.readout = nn.Linear(hidden_size, max_values)

    def forward(self, inputs, positions, value_counts):
        if any(k in inputs for k in ('labels','past_key_values','use_cache','inputs_embeds')):
            raise ValueError('label-free, uncached input_ids required')
        ids = inputs['input_ids']; mask = inputs['attention_mask']
        if positions.dtype != torch.long or positions.ndim != 2 or positions.shape[0] != ids.shape[0]:
            raise ValueError('batched integer span positions required')
        if value_counts.shape != positions.shape or value_counts.dtype != torch.long:
            raise ValueError('aligned integer value counts required')
        if ((positions < 0) | (positions >= ids.shape[1])).any():
            raise ValueError('span outside input sequence')
        if positions.shape[1] > 1 and (positions[:,1:] <= positions[:,:-1]).any():
            raise ValueError('span positions must increase')
        if not mask.gather(1, positions).bool().all():
            raise ValueError('span points into padding')
        if ((value_counts < 2) | (value_counts > self.readout.out_features)).any():
            raise ValueError('value count exceeds head capacity')
        hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
        selected = hidden.gather(1, positions.unsqueeze(-1).expand(-1,-1,hidden.shape[-1]))
        logits = self.readout(selected).float()
        valid = torch.arange(logits.shape[-1],device=logits.device) < value_counts.unsqueeze(-1)
        return logits.masked_fill(~valid, -torch.inf)


def span_supervised_loss(logits, targets, supervised):
    """Unknown semantic values are classes; missing supervision is only a mask."""
    if targets.shape != logits.shape[:2] or supervised.shape != targets.shape:
        raise ValueError('aligned per-span targets and supervision required')
    if targets.dtype != torch.long or supervised.dtype != torch.bool or not supervised.any():
        raise ValueError('nonempty boolean supervision and integer targets required')
    selected = targets[supervised]
    if ((selected < 0) | (selected >= logits.shape[-1])).any():
        raise ValueError('supervised target outside classes')
    active = logits[supervised]
    if not torch.isfinite(active.gather(1,selected[:,None])).all():
        raise ValueError('supervised target is a masked class')
    return nn.functional.cross_entropy(active, selected)
