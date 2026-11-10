"""Read-only single-row Omni text-layer hooks for padding diagnostics.

Reference tensors are stored one event per file; comparison loads one at a
time. Inference only, no retained autograd graph, no returned replacement
activations. A trace does not establish model/source/input identity by itself.
"""
from contextlib import contextmanager
from pathlib import Path

from .baseline import file_hash


class PaddingTrace:
    def __init__(self, directory, *, max_bytes=8*1024**3):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive storage bound required')
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.max_bytes=max_bytes
        self.bytes=0
        self.reference={}
        self.reports=[]
        self.active=False
        self.complete=False

    @contextmanager
    def capture(self, text_model, attention_mask, *, compare=False):
        import torch
        if self.active or text_model.training or torch.is_grad_enabled():
            raise ValueError('non-nested eval inference tracing required')
        if (attention_mask.ndim!=2 or attention_mask.shape[0]!=1
                or not torch.all((attention_mask==0)|(attention_mask==1))
                or not attention_mask.any()):
            raise ValueError('single-row nonempty binary token mask required')
        if compare != self.complete or (not compare and self.reference):
            raise ValueError('one complete reference required before comparison')
        mask=attention_mask[0].bool().cpu().clone()
        width=mask.numel(); seen=[]; handles=[]; expected=[]
        reports=[]

        def observe(name, value, *, flat=False, indices=False):
            if name in seen:raise ValueError('duplicate hook event')
            if not isinstance(value,torch.Tensor):raise ValueError('tensor hook output required')
            if flat:
                if value.ndim!=2 or value.shape[0]!=width:raise ValueError('router shape differs')
                valid=value.detach().cpu()[mask].clone()
            else:
                if value.ndim!=3 or value.shape[:2]!=(1,width):raise ValueError('activation shape differs')
                valid=value.detach().cpu()[0,mask].clone()
            if not torch.isfinite(valid).all():raise ValueError('nonfinite valid activation')
            seen.append(name)
            path=self.directory/(name+'.pt')
            if not compare:
                size=valid.numel()*valid.element_size()
                if self.bytes+size>self.max_bytes:raise ValueError('reference tensor storage budget exceeded')
                self.bytes+=size
                torch.save(valid,path)
                self.reference[name]=dict(sha256=file_hash(path),shape=list(valid.shape),dtype=str(valid.dtype))
            else:
                pin=self.reference[name]
                if file_hash(path)!=pin['sha256']:raise ValueError('reference tensor changed')
                prior=torch.load(path,map_location='cpu',weights_only=True)
                if (prior.shape!=valid.shape or prior.dtype!=valid.dtype
                        or list(prior.shape)!=pin['shape'] or str(prior.dtype)!=pin['dtype']):
                    raise ValueError('valid tensor identity differs')
                item=dict(event=name,equal=bool(torch.equal(prior,valid)),
                          differing_elements=int((prior!=valid).sum()))
                if indices:
                    item.update(ordered_route_changed_tokens=int((prior!=valid).any(-1).sum()),
                        expert_set_changed_tokens=int((prior.sort(-1).values!=valid.sort(-1).values).any(-1).sum()))
                else:
                    delta=(prior.float()-valid.float()).abs()
                    item.update(max_abs_error=float(delta.max()),mean_abs_error=float(delta.mean()))
                reports.append(item)

        def pre_hook(name):
            def hook(module,args,kwargs):
                observe(name,args[0] if args else kwargs['hidden_states'])
            return hook

        def out_hook(name,tuple_first=False):
            def hook(module,args,output):
                observe(name,output[0] if tuple_first else output)
            return hook

        def router_hook(prefix):
            def hook(module,args,output):
                if not isinstance(output,tuple) or len(output)!=3:raise ValueError('three router outputs required')
                for suffix,value in zip(('logits','weights','indices'),output,strict=True):
                    observe(prefix+'.'+suffix,value,flat=True,indices=suffix=='indices')
            return hook

        self.active=True
        try:
            for i,layer in enumerate(text_model.layers):
                prefix=f'layer-{i:02d}'
                expected.extend([prefix+'.input',prefix+'.attention'])
                handles.append(layer.register_forward_pre_hook(pre_hook(prefix+'.input'),with_kwargs=True))
                handles.append(layer.self_attn.register_forward_hook(out_hook(prefix+'.attention',True)))
                if hasattr(layer.mlp,'gate'):
                    expected.extend(prefix+'.router.'+s for s in ('logits','weights','indices'))
                    handles.append(layer.mlp.gate.register_forward_hook(router_hook(prefix+'.router')))
                expected.extend([prefix+'.mlp',prefix+'.output'])
                handles.append(layer.mlp.register_forward_hook(out_hook(prefix+'.mlp')))
                handles.append(layer.register_forward_hook(out_hook(prefix+'.output')))
            if not expected:raise ValueError('nonempty text layers required')
            yield
            if seen!=expected or (compare and seen!=list(self.reference)):
                raise ValueError('incomplete ordered layer coverage')
            if compare:self.reports=reports
            else:self.complete=True
        finally:
            for handle in handles:handle.remove()
            self.active=False
