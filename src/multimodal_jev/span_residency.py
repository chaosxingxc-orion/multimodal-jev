"""Explicit FSDP pause/resume boundary; caller owns collective/device leases.

Callbacks must use the pinned backend's FSDP-aware model movement. This is not
ordinary model.to(), does not free buffers/communication allocations, and does
not authorize a concurrent inference process. Failure requires exact recovery.
"""


class SpanResidency:
    def __init__(self, model, optimizer, *, offload_model, load_model, synchronize):
        self.model, self.optimizer = model, optimizer
        self.offload_model, self.load_model = offload_model, load_model
        self.synchronize = synchronize
        self.phase = 'resident'

    def _identity(self):
        return ([id(p) for p in self.model.parameters()],
                [[id(p) for p in g['params']] for g in self.optimizer.param_groups])

    def offload(self):
        import torch
        if self.phase != 'resident': raise ValueError('resident boundary required')
        if any(p.grad is not None for p in self.model.parameters()):
            raise ValueError('clear gradients at a completed optimizer boundary')
        self.phase = 'offloading'
        try:
            self.synchronize()
            self.identities = self._identity()
            self.parameter_devices = {n:p.device for n,p in self.model.named_parameters()}
            self.parameter_values = {n:p.detach().cpu().clone() for n,p in self.model.named_parameters()}
            self.state_devices = {(p,k):v.device for p,s in self.optimizer.state.items()
                                  for k,v in s.items() if isinstance(v,torch.Tensor)}
            self.state_values = {(p,k):v.detach().cpu().clone() for p,s in self.optimizer.state.items()
                                 for k,v in s.items() if isinstance(v,torch.Tensor)}
            self.offload_model(self.model)
            for (p,k) in self.state_devices:
                self.optimizer.state[p][k] = self.optimizer.state[p][k].to('cpu', non_blocking=True)
            self.synchronize()
            self._verify(cpu=True)
            self.phase = 'offloaded'
        except BaseException:
            self.phase = 'failed'; raise

    def _verify(self, *, cpu):
        import torch
        if self.identities != self._identity(): raise ValueError('parameter or optimizer references changed')
        actual = dict(self.model.named_parameters())
        if actual.keys() != self.parameter_values.keys(): raise ValueError('parameter layout changed')
        for n,p in actual.items():
            expected_device = torch.device('cpu') if cpu else self.parameter_devices[n]
            if (p.device != expected_device or p.dtype != self.parameter_values[n].dtype
                    or not torch.equal(p.detach().cpu(),self.parameter_values[n])):
                raise ValueError('parameter placement or value changed: '+n)
        keys = {(p,k) for p,s in self.optimizer.state.items() for k,v in s.items() if isinstance(v,torch.Tensor)}
        if keys != self.state_devices.keys(): raise ValueError('optimizer state layout changed')
        for p,k in keys:
            v = self.optimizer.state[p][k]
            expected_device = torch.device('cpu') if cpu else self.state_devices[p,k]
            if (v.device != expected_device or v.dtype != self.state_values[p,k].dtype
                    or not torch.equal(v.detach().cpu(),self.state_values[p,k])):
                raise ValueError('optimizer placement or value changed: '+k)

    def reload(self):
        if self.phase != 'offloaded': raise ValueError('verified offload required')
        self.phase = 'loading'
        try:
            self._verify(cpu=True)
            self.load_model(self.model)
            # Adam's CPU step tensor must remain CPU even when moments live on
            # NPU. Preserve each original tensor device instead of moving all.
            for (p,k),device in self.state_devices.items():
                self.optimizer.state[p][k] = self.optimizer.state[p][k].to(device, non_blocking=True)
            self.synchronize()
            self._verify(cpu=False)
            self.phase = 'resident'
            del self.parameter_values, self.state_values
        except BaseException:
            self.phase = 'failed'; raise
