"""Opt-in logits alongside unchanged probabilities; no additional head forward.

Install only on a separately versioned, quiescent worker. Preserve the readout
object (activation transactions hold it) and all registered parameter names.
"""
from types import MethodType
import torch

WIRE = 'context-prob-logits-fp32-v1'


def install_logits_readout(readout):
    if getattr(readout, '_jev_logits_wire', None) is not None:
        raise ValueError('logits wire already installed')
    original = readout.forward
    if not hasattr(readout.head, 'projection'):
        raise ValueError('shared head with scalar projection required')
    readout._jev_logits_busy = False

    def forward(self, hidden, lengths, prompts, extras):
        if self._jev_logits_busy:
            raise ValueError('concurrent readout forbidden')
        self._jev_logits_busy = True
        captured = []
        def capture(module, args, output):
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[-1] != 1:
                raise ValueError('one complete row scalar head projection required')
            captured.append(output[0].float())
        handle = self.head.projection.register_forward_hook(capture)
        try:
            probabilities = original(hidden, lengths, prompts, extras)
            if len(captured) != len(probabilities):
                raise ValueError('exactly one actual projection per complete request required')
            outputs = []
            for p,z in zip(probabilities,captured,strict=True):
                if (p.dtype != torch.float32 or z.shape != p.shape or not torch.isfinite(z).all()
                        or not torch.equal(z.sigmoid(),p)):
                    raise ValueError('actual projection/logit/probability path differs')
                outputs.append(torch.cat((p,z),dim=-1))
            return outputs
        finally:
            handle.remove(); self._jev_logits_busy = False

    readout.forward = MethodType(forward,readout)
    readout._jev_logits_wire = WIRE
    return dict(wire=WIRE,columns=['prob','logit'],extra_head_forwards=0)


def validate_logits_record(record):
    if record.get('output_protocol') != WIRE or not isinstance(record.get('logits'),list):
        raise ValueError('actual logits protocol required')
    p=torch.tensor(record['prob'],dtype=torch.float32)
    z=torch.tensor(record['logits'],dtype=torch.float32)
    if (p.ndim!=1 or p.numel()<1 or z.shape!=p.shape or not torch.isfinite(z).all()
            or not torch.isfinite(p).all() or ((p<=0)|(p>=1)).any()):
        raise ValueError('complete finite interior probability/logit vectors required')
    # CPU recomputation is only a corruption bound; NPU producer asserts exact
    # sigmoid identity on-device. Do not replace the actual probability by it.
    if (z.sigmoid()-p).abs().max().item()>2e-7:
        raise ValueError('logit and probability incompatible')
    return record
