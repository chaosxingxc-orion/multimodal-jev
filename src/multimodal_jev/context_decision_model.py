"""One audio backbone invocation followed by the shared probability branch."""
import torch
from torch import nn


class ContextDecisionModel(nn.Module):
    """Caller installs backbone LoRA and token embedding before construction.

    This module owns no LM vocabulary head and never invokes generation. A fixed
    token ID is required; tokenizer/config/weights must be bound by policy export.
    """
    def __init__(self, backbone, decision_head, decision_token_id):
        super().__init__()
        if type(decision_token_id) is not int or decision_token_id < 0:
            raise ValueError('registered decision token ID required')
        self.backbone = backbone
        self.decision_head = decision_head
        self.decision_token_id = decision_token_id

    def score_logits(self, inputs, positions, *, return_marker=False, return_speech=False):
        if any(k in inputs for k in ('labels', 'past_key_values', 'use_cache', 'inputs_embeds',
                                     'return_dict', 'output_hidden_states')):
            raise ValueError('label-free uncached explicit token inputs required')
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        if ids.ndim != 2 or ids.dtype != torch.long or mask.shape != ids.shape:
            raise ValueError('batched token IDs and aligned mask required')
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError('binary attention mask required')
        if self.decision_token_id==151706 and (ids==151705).any():
            raise ValueError('timestamp reservation cannot appear in decision input')
        if (positions.ndim != 2 or positions.dtype != torch.long or positions.shape[0] != ids.shape[0]
                or positions.shape[1] == 0 or positions.device != ids.device
                or ((positions < 0) | (positions >= ids.shape[1])).any()):
            raise ValueError('valid batched marker positions required')
        if positions.shape[1] > 1 and (positions[:, 1:] <= positions[:, :-1]).any():
            raise ValueError('ordered distinct marker positions required')
        if (not mask.gather(1, positions).bool().all()
                or not (ids.gather(1, positions) == self.decision_token_id).all()
                or not (((ids == self.decision_token_id) & mask.bool()).sum(-1) == positions.shape[1]).all()):
            raise ValueError('positions must cover every valid decision marker exactly once')
        if return_speech and 'audio_owner_tokens' not in inputs:
            raise ValueError('speech contrast requires the shared frozen acoustic forward')
        if 'audio_owner_tokens' in inputs:
            from .card_training_acoustics import hidden as acoustic_hidden
            hidden = acoustic_hidden(self.backbone, inputs, return_speech=return_speech)
            if return_speech:
                hidden, speech = hidden
        else:
            hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
        # Enter the head's Module.__call__ as well: its own FSDP unit must
        # gather/cast all attention, normalization and FFN parameters together.
        if return_speech:
            logits, vectors = self.decision_head(hidden, positions, mask.bool(), return_logits=True, speech=speech)
            return logits, vectors.index_select(0, inputs['speech_row_owners'])
        logits = self.decision_head(hidden, positions, mask.bool(), return_logits=True)
        if return_marker:
            return logits, hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
        return logits

    def forward(self, inputs, positions, *, return_logits=False, return_marker=False, return_speech=False):
        if type(return_logits) is not bool:
            raise ValueError('explicit Boolean logits mode required')
        if return_speech:
            if not return_logits or return_marker:
                raise ValueError('speech contrast requires logits and independent audio, not markers')
            return self.score_logits(inputs, positions, return_speech=True)
        if return_marker:
            if not return_logits:
                raise ValueError("marker training requires the membership logits")
            return self.score_logits(inputs, positions, return_marker=True)
        logits = self.score_logits(inputs, positions)
        return logits if return_logits else logits.sigmoid()
