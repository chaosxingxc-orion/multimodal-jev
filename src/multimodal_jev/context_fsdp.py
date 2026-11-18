"""FSDP units for the batched context model, installed before wrapping.

MultiheadAttention invokes its output projection functionally; wrapping that
Linear alone would bypass its FSDP pre-forward hook. Keep the complete shared
head in one unit, and retain the established backbone Linear units.
"""
from .span_fsdp import unshared_linear_modules


def context_fsdp_units(model):
    head=model.decision_head
    descendants=set(head.modules())
    head_parameters={id(p) for p in head.parameters()}
    if not head_parameters or head_parameters & {id(p) for p in model.backbone.parameters()}:
        raise ValueError('nonempty head with parameters distinct from backbone required')
    return (unshared_linear_modules(model)-descendants)|{head}
