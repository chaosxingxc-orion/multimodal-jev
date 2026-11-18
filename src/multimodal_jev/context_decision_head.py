"""Candidate history-attending Jev branch; not a production backend adapter.

Inputs are contextual backbone states from one complete request per batch row.
Only selected marker positions are queries; K/V include the valid causal prefix.
Position information is inherited from the backbone, with no additional branch
position embedding in this version. Final cannot consume same-layer outputs.
"""
import torch
from torch import nn


class ContextDecisionHead(nn.Module):
    """One shared Transformer branch yielding independent applicability prob.

    No candidate-specific parameters, categorical normalization or text decode.
    Caller supplies marker-self positions after compiling the actual audio input.
    Token registration, policy export and vLLM loading are separate obligations.
    """

    def __init__(self, hidden_size, num_heads, ffn_size, residual_scale=0.1):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (hidden_size, num_heads, ffn_size)):
            raise ValueError('positive integer dimensions required')
        if hidden_size % num_heads:
            raise ValueError('hidden size must be divisible by heads')
        if not 0 < residual_scale <= 1:
            raise ValueError('nonzero residual scale in (0, 1] required')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.residual_scale = float(residual_scale)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, ffn_size), nn.GELU(),
                                 nn.Linear(ffn_size, hidden_size))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, 1)

    def _validate(self, hidden, positions, valid_tokens):
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size or not hidden.is_floating_point():
            raise ValueError('floating [batch, length, hidden] states required')
        b, length, _ = hidden.shape
        if b == 0 or length == 0:
            raise ValueError('nonempty requests required')
        if (positions.dtype != torch.long or positions.ndim != 2 or
                positions.shape[0] != b or positions.shape[1] == 0):
            raise ValueError('nonempty batched integer marker positions required')
        if valid_tokens.dtype != torch.bool or valid_tokens.shape != (b, length):
            raise ValueError('boolean valid-token mask required')
        if positions.device != hidden.device or valid_tokens.device != hidden.device:
            raise ValueError('inputs must share device')
        if ((positions < 0) | (positions >= length)).any():
            raise ValueError('marker outside sequence')
        if positions.shape[1] > 1 and (positions[:, 1:] <= positions[:, :-1]).any():
            raise ValueError('ordered distinct markers required')
        if not valid_tokens.gather(1, positions).all():
            raise ValueError('marker in padding')
        if not torch.isfinite(hidden).all():
            raise ValueError('finite hidden states required, including padding')

    def score_logits(self, hidden, positions, valid_tokens):
        """Internal logits for stable proper-scoring/RL losses, not success labels."""
        self._validate(hidden, positions, valid_tokens)
        b, length, width = hidden.shape
        indices = positions.unsqueeze(-1).expand(-1, -1, width)
        selected = hidden.gather(1, indices)
        context = self.context_norm(hidden)
        query = context.gather(1, indices)
        forbidden = ((torch.arange(length, device=hidden.device)[None, None, :] > positions[:, :, None])
                     | ~valid_tokens[:, None, :])
        mask = forbidden[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            b * self.num_heads, positions.shape[1], length)
        attended, _ = self.attention(query, context, context, attn_mask=mask, need_weights=False)
        state = selected + self.residual_scale * attended
        state = state + self.residual_scale * self.ffn(self.ffn_norm(state))
        logits = self.projection(self.output_norm(state)).squeeze(-1)
        # BF16 execution with FP32 probability/loss arithmetic; preserve FP64 diagnostics.
        return logits.float() if logits.dtype in (torch.bfloat16, torch.float16) else logits

    def forward(self, hidden, positions, valid_tokens, *, return_logits=False):
        """Return [batch, decisions] applicability probabilities in [0, 1]."""
        if type(return_logits) is not bool:
            raise ValueError('explicit Boolean logits mode required')
        logits = self.score_logits(hidden, positions, valid_tokens)
        return logits if return_logits else logits.sigmoid()
