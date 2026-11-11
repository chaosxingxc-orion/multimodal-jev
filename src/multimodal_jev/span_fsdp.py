"""Conservative FSDP unit selection for shared-weight native Span models."""
def unshared_linear_modules(model):
    """Keep shared parameters in the common enclosing/root FSDP unit.

    In Qwen ASR the LM head aliases token embeddings. Wrapping that Linear
    alone lets the embedding observe a local 1-D shard before all-gather.
    This policy selects only Linear modules with no cross-module aliases.
    """
    import torch
    owners={}
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter),set()).add(id(module))
    return {module for module in model.modules() if isinstance(module,torch.nn.Linear)
            and all(len(owners[id(p)])==1 for p in module.parameters(recurse=False))}
