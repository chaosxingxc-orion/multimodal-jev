"""Translate a Thinker audio policy to PEFT names consumed by vLLM."""
import re

from .context_policy_artifacts import peft_payload


def thinker_target(name):
    if not re.fullmatch(r'thinker\.model\.layers\.\d+\.self_attn\.[qv]_proj', name):
        raise ValueError('expected a Thinker text q/v LoRA target: ' + name)
    return name


def thinker_payload(bundle):
    return peft_payload(bundle, thinker_target)


def packed_targets(specs):
    return sorted({
        thinker_target(name).replace('thinker.model.', 'language_model.model.', 1)
        .rsplit('.', 1)[0] + '.qkv_proj'
        for name in specs
    })
