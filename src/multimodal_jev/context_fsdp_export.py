"""Validate an assembled FSDP execution payload before loading a plain model.

Caller authenticates the training checkpoint, base, processor and marker row
separately. This bridge never guesses missing head weights or casts FP32 masters.
"""
import torch

from .context_policy_snapshot import _parameters


def canonical_execution(tensors):
    if type(tensors) is not dict or not tensors:
        raise ValueError('complete assembled BF16 execution map required')
    result={}
    for name,value in tensors.items():
        if type(name) is not str or not name:
            raise ValueError('named FSDP execution tensors required')
        # Only strip actual wrapper path components, never substrings.
        key='.'.join(part for part in name.split('.') if part!='_fsdp_wrapped_module')
        if key in result:
            raise ValueError('ambiguous FSDP execution key: '+key)
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.bfloat16 or
                value.device.type!='cpu' or not torch.isfinite(value).all()):
            raise ValueError('finite assembled CPU BF16 tensor required: '+name)
        result[key]=value
    return result


def load_assembled_execution(model,tensors):
    saved=canonical_execution(tensors)
    expected=_parameters(model)
    if saved.keys()!=expected.keys():
        raise ValueError('complete head and LoRA execution required')
    for name,param in expected.items():
        if saved[name].shape!=param.shape:
            raise ValueError('assembled execution shape differs: '+name)
    with torch.no_grad():
        for name,param in expected.items():param.copy_(saved[name])
    if any(not torch.equal(param.detach().cpu(),saved[name]) for name,param in expected.items()):
        raise ValueError('assembled execution load differs')
    return dict(parameters=len(expected),exact_bf16=True,optimizer_restored=False)
