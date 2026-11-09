"""Persist optimizer coordinates for a fixed-base LoRA actor."""
from pathlib import Path


def trainable_parameters(model):
    parameters=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('no trainable actor parameters')
    return parameters


def save_optimizer(model,optimizer,path):
    import torch
    path=Path(path)
    parameters=trainable_parameters(model)
    ordered=[p for group in optimizer.param_groups for p in group['params']]
    if len(ordered)!=len(parameters) or any(a is not b for a,(_,b) in zip(ordered,parameters)):
        raise ValueError('optimizer/model parameter order mismatch')
    state={'optimizer':optimizer.state_dict(),'parameter_names':[name for name,_ in parameters],
           'parameter_shapes':[list(p.shape) for _,p in parameters],'cpu_rng':torch.get_rng_state()}
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary);temporary.replace(path)


def restore_optimizer(model,optimizer,path,device='cpu'):
    import torch
    state=torch.load(path,map_location=device,weights_only=True)
    parameters=trainable_parameters(model)
    if state['parameter_names']!=[name for name,_ in parameters]:
        raise ValueError('optimizer parameter mapping changed')
    if state['parameter_shapes']!=[list(p.shape) for _,p in parameters]:
        raise ValueError('optimizer parameter shapes changed')
    ordered=[p for group in optimizer.param_groups for p in group['params']]
    if len(ordered)!=len(parameters) or any(a is not b for a,(_,b) in zip(ordered,parameters)):
        raise ValueError('optimizer/model parameter order mismatch')
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'].cpu())
