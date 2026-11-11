"""BF16 execution parameters with FP32 master AdamW for the Span pilot.

This local optimizer bridge is not a distributed verl optimizer integration.
Exported/served parameters are exactly the BF16 execution copy.
"""
import torch


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
                    scope='Local pilot; FP32 master/AdamW state, BF16 execution/export; no resume support.')
