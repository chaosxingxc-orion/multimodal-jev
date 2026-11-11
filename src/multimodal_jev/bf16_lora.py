"""BF16 execution parameters with FP32 master AdamW for the Span pilot.

This local optimizer bridge is not a distributed verl optimizer integration.
Exported/served parameters are exactly the BF16 execution copy.
"""
import torch
import copy


def cast_lora_bf16(model):
    found = []
    for name, parameter in model.named_parameters():
        if 'lora_A.' in name or 'lora_B.' in name:
            if parameter.grad is not None:
                raise ValueError('cast adapters before backward/optimizer construction')
            parameter.data = parameter.data.to(torch.bfloat16)
            found.append(name)
        elif parameter.requires_grad:
            raise ValueError('BF16 pilot permits only LoRA A/B training')
    if not found:
        raise ValueError('no LoRA A/B parameters')
    return found


class BF16MasterAdamW:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        if not self.parameters or any(p.dtype != torch.bfloat16 or not p.requires_grad
                                      for p in self.parameters):
            raise ValueError('trainable BF16 execution parameters required')
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=0.,
                                          foreach=False, fused=False)

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        # Caller clips execution gradients before copying them to FP32 masters.
        for p, master in zip(self.parameters, self.masters, strict=True):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ValueError('nonfinite BF16 adapter gradient')
            master.grad = None if p.grad is None else p.grad.detach().float().clone()
        self.optimizer.step()
        for p, master in zip(self.parameters, self.masters, strict=True):
            if not torch.isfinite(master).all():
                raise ValueError('nonfinite FP32 master parameter')
            p.copy_(master)

    def receipt(self):
        return dict(execution_parameter_dtypes=sorted({str(p.dtype) for p in self.parameters}),
                    master_parameter_dtypes=sorted({str(p.dtype) for p in self.masters}),
                    optimizer_tensor_dtypes=sorted({str(v.dtype) for state in self.optimizer.state.values()
                                                   for v in state.values() if torch.is_tensor(v)}),
                    parameters=len(self.parameters),
                    optimizer_state_parameters=len(self.optimizer.state),
                    optimizer_steps=sorted({int(state['step'].item()) for state in self.optimizer.state.values()
                                            if 'step' in state}),
                    scope='Local pilot; FP32 master/AdamW state, BF16 execution/export; ordered-parameter resume.')

    def state_dict(self):
        return {'schema':'bf16-master-adamw-v1',
                'masters':[p.detach().cpu().clone() for p in self.masters],
                'optimizer':copy.deepcopy(self.optimizer.state_dict())}

    def load_state_dict(self, state):
        """Restore only after loading the matching BF16 execution checkpoint.

Caller must bind ordered parameter names and model identity in its checkpoint
manifest. Shape checks alone cannot authenticate two same-shaped parameters.
"""
        if state.get('schema')!='bf16-master-adamw-v1' or len(state['masters'])!=len(self.parameters):
            raise ValueError('incompatible master checkpoint')
        for p,master in zip(self.parameters,state['masters'],strict=True):
            if (master.dtype!=torch.float32 or master.shape!=p.shape or not torch.isfinite(master).all()
                    or not torch.equal(master.to(device=p.device,dtype=p.dtype),p.detach())):
                raise ValueError('master checkpoint differs from loaded BF16 execution parameters')
        with torch.no_grad():
            for master,saved in zip(self.masters,state['masters'],strict=True):
                master.copy_(saved)
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.zero_grad()
