"""Dual-credit wire atop the existing bounded bulk/segmented engine.

No new scheduler or activation implementation. Engine construction remains a
finite candidate pending real NPU parity and training admission.
"""
import copy
from types import SimpleNamespace

import torch

from multimodal_jev.context_logits_wire import WIRE,validate_logits_record
from tools.reports.context_batched_request_candidate import prepare_request,encode_many
from tools.reports.context_batched_owner_candidate import transformed_source,PROFILE
from tools.reports.context_bulk_rpc_candidate import BulkContextEngineClient

RUNTIME = 'context-segmented-dual-logits-candidate-v1'


def encode_logits_many(owner,rows,*,audio_token_id,params_class):
    """Reuse exact batch input validation/encode/order checking; add a column."""
    raw_columns = []
    engine = owner.engine
    class ProbabilityView:
        def encode(self,*args,**kwargs):
            results=engine.encode(*args,**kwargs)
            views=[]
            for result in results:
                value=result.outputs.data
                if (not isinstance(value,torch.Tensor) or value.ndim!=2 or value.shape[1]!=2
                        or value.dtype!=torch.float32 or not torch.isfinite(value).all()):
                    raise ValueError('worker must return actual FP32 probability/logit columns')
                raw_columns.append(value[:,1].detach().cpu().tolist())
                views.append(SimpleNamespace(prompt_token_ids=result.prompt_token_ids,
                    outputs=SimpleNamespace(data=value[:,:1])))
            return views
    # Separate facade avoids mutating the live owner's engine during a request.
    facade=SimpleNamespace(bundle=owner.bundle,max_batch_queries=owner.max_batch_queries,
        max_batch_tokens=owner.max_batch_tokens,max_model_len=owner.max_model_len,
        max_decisions=owner.max_decisions,lora=owner.lora,engine=ProbabilityView())
    policy=owner.bundle['policy_id']
    records=encode_many(facade,rows,audio_token_id=audio_token_id,params_class=params_class)
    if owner.bundle['policy_id']!=policy:
        raise ValueError('policy switched during logits batch')
    for record,z in zip(records,raw_columns,strict=True):
        record.update(logits=z,output_protocol=WIRE)
        validate_logits_record(record)
    return records


class DualContextEngineClient(BulkContextEngineClient):
    def score_many(self,rows,*,audio_token_id):
        try:
            values=super().score_many(rows,audio_token_id=audio_token_id)
            for record in values:validate_logits_record(record)
            return values
        except BaseException:
            self.phase='failed';raise


def owner_class(source_root,*,shared_scope=False,scope_module=None):
    source=transformed_source(source_root)
    if scope_module is not None:
        from pathlib import Path
        import hashlib
        name=scope_module.get('name') if isinstance(scope_module,dict) else None
        if (not isinstance(name,str) or not name.isidentifier() or name.startswith('_')
                or set(scope_module)!={'name','sha256'}):
            raise ValueError('explicit scope module name/hash required')
        path=Path(source_root)/'tools/jev'/(name+'.py')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=scope_module['sha256']:
            raise ValueError('configurable physical device validator changed')
        before='from multimodal_jev.span_device_scope import validate_tp1'
        if source.count(before)!=1:raise ValueError('device scope binding changed')
        source=source.replace(before,'from '+name+' import validate_tp1')
    elif shared_scope:
        from pathlib import Path
        import hashlib
        path=Path(source_root)/'tools/jev/context_shared_device_scope.py'
        if hashlib.sha256(path.read_bytes()).hexdigest()!='884ef81232750424d4a0bf8a105683dd99988fbec1c9b21259c50df18377ddbb':
            raise ValueError('shared physical device validator changed')
        before='from multimodal_jev.span_device_scope import validate_tp1'
        if source.count(before)!=1:raise ValueError('device scope binding changed')
        source=source.replace(before,'from context_shared_device_scope import validate_tp1')
    before='tools.reports.context_batched_worker_extension_candidate.BatchedContextWorkerCandidate'
    if source.count(before)!=1:raise ValueError('worker binding changed')
    source=source.replace(before,'tools.reports.context_dual_worker_candidate.DualContextWorkerCandidate')
    namespace={'__name__':__name__+'.derived'}
    exec(compile(source,'<dual-segmented-owner>', 'exec'),namespace)
    base=namespace['SwitchingContextRollout']
    class DualContextOwner(base):
        def __init__(self,config):
            if (config.get('candidate_runtime')!=RUNTIME or config.get('output_protocol')!=WIRE
                    or type(config.get('max_batch_queries')) is not int
                    or not 1<=config['max_batch_queries']<=128
                    or type(config.get('max_batch_tokens')) is not int
                    or config['max_batch_tokens']<config['max_model_len']):
                raise ValueError('explicit bounded dual runtime required')
            self.max_batch_queries=config['max_batch_queries'];self.max_batch_tokens=config['max_batch_tokens']
            super().__init__(config)
            try:
                receipt=self.engine.llm_engine.collective_rpc('install_jev_dual_candidate',
                                                            args=(self.bundle['policy_id'],))
                if (len(receipt)!=1 or receipt[0]['policy_id']!=self.bundle['policy_id']
                        or receipt[0]['wire']!=WIRE or receipt[0]['training_dispatch_allowed']
                        or receipt[0]['layers']<=0):
                    raise ValueError('dual worker installation failed')
                self.segmented_receipt=receipt[0]
            except BaseException:self.close();raise

        def score_many(self,rows,*,audio_token_id):
            from vllm import PoolingParams
            with self._transaction:
                if self.phase!='ready':raise ValueError('ready quiescent owner required')
                self.phase='evaluating'
                try:values=encode_logits_many(self,rows,audio_token_id=audio_token_id,params_class=PoolingParams)
                except BaseException:self.phase='failed';raise
                self.phase='ready';return values

        def score(self,inputs,positions,*,audio_token_id):
            return self.score_many([dict(inputs=inputs,positions=positions)],audio_token_id=audio_token_id)[0]
    return DualContextOwner
