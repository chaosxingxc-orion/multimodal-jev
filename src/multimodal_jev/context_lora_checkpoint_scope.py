"""Reject backbone tensors in LoRA/head-only training checkpoints."""


def validate_trainable_scope(named_parameters):
    named=list(named_parameters)
    if not named:raise ValueError('nonempty LoRA/head checkpoint scope required')
    counts=dict(lora=0,head=0)
    for name,p in named:
        key='.'.join(part for part in name.split('.') if part not in ('_fsdp_wrapped_module','module'))
        if not p.requires_grad:raise ValueError('frozen tensor in training checkpoint: '+name)
        if key.startswith('decision_head.'):
            counts['head']+=p.numel()
        elif key.startswith('backbone.') and ('.lora_A.' in key or '.lora_B.' in key):
            counts['lora']+=p.numel()
        else:raise ValueError('base model tensor forbidden in LoRA checkpoint: '+name)
    # FSDP can expose empty local views; do not require each rank to own both.
    return dict(tensors=len(named),local_numel=counts,base_model_saved=False,
                scope='LoRA and Jev head only; base identity referenced separately')
