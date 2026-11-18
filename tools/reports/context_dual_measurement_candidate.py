"""Finite dual-wire serial/batch comparison using the original measurement loop.

Serial uses the unchanged single-request backbone profile; only its readout
wire exposes actual logits. Batched uses the separately versioned segmented
profile. No labels, sampling rewards, optimizer or target selection here.
"""
import argparse
import json
from pathlib import Path
from types import MethodType

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_logits_wire import install_logits_readout,validate_logits_record
from tools.reports.context_batch_measurement_candidate import (
    measure,compare,install_counters,reset_counters,read_counters)
from tools.reports.context_dual_rollout_candidate import owner_class,encode_logits_many


def serial_owner_class(source_root):
    """Keep the original single-sequence backbone; add named measurement RPC."""
    import hashlib
    from tools.reports.context_batched_owner_candidate import SOURCE_SHA
    path=Path(source_root)/'tools/jev/context_switch_rollout.py'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise ValueError('frozen serial owner changed')
    before="worker_extension_cls='context_switch_worker.SwitchingContextWorker',"
    source=raw.decode()
    if source.count(before)!=1:raise ValueError('serial worker binding changed')
    source=source.replace(before,"worker_extension_cls='tools.reports.context_dual_worker_candidate.DualContextWorkerCandidate',")
    namespace={'__name__':__name__+'.serial'}
    exec(compile(source,'<serial-measurement-owner>','exec'),namespace)
    return namespace['SwitchingContextRollout']


def install_serial_wire(model):
    if model.pooler.memory_profiling or not model._jev_base_weights_loaded:
        raise ValueError('actual loaded quiescent model required')
    result=install_logits_readout(model.pooler.readout)
    return dict(result,policy_id=model._jev_policy_id)


def attach_serial_score(owner):
    owner.max_batch_queries=1
    owner.max_batch_tokens=owner.max_model_len
    def score(self,inputs,positions,*,audio_token_id):
        from vllm import PoolingParams
        with self._transaction:
            if self.phase!='ready':raise ValueError('ready serial owner required')
            self.phase='evaluating'
            try:
                values=encode_logits_many(self,[dict(inputs=inputs,positions=positions)],
                    audio_token_id=audio_token_id,params_class=PoolingParams)
            except BaseException:self.phase='failed';raise
            self.phase='ready';return values[0]
    owner.score=MethodType(score,owner)


def validate_measurement(result):
    # Complete actual logits are checked for every pass, including warmup.
    for item in result['passes']:
        for record in item['records']:validate_logits_record(record)
    return result


def compare_dual(serial,batched):
    validate_measurement(serial);validate_measurement(batched)
    result=compare(serial,batched) # full record equality includes actual logits
    return dict(result,exact_actual_logits=True,
        scope='Fixed-policy TRAIN128 serial versus batch, actual logits/probability and original Bernoulli G16; dual ranking sample groups and full TRAIN coverage separate')


def audit_dual_measurements(serial,batched,rows,policy_id,input_sha256,max_queries,max_tokens):
    from tools.reports.audit_context_batch_probe_candidate import audit_measurements
    validate_measurement(serial);validate_measurement(batched)
    result=audit_measurements(serial,batched,rows,policy_id,input_sha256,max_queries,max_tokens)
    return dict(result,exact_actual_logits=True,
        scope='Fixed-policy finite inference and original Bernoulli G16; no dual-ranking sampling, full TRAIN saturation, actor update or resume claim')


def run(config,output,mode):
    import torch
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json
    from tools.reports.context_bulk_rpc_candidate import validate_rows
    output=Path(output);require_training_storage(output);output.mkdir(exist_ok=False)
    for name,digest in config['pins'].items():
        if file_hash(name)!=digest:raise ValueError('measurement dependency changed: '+name)
    if file_hash(config['inputs'])!=config['input_sha256']:
        raise ValueError('fixed input payload changed')
    rows=validate_rows(torch.load(config['inputs'],map_location='cpu',weights_only=True))
    if len(rows)!=128 or mode not in ('serial','batched'):
        raise ValueError('fixed TRAIN128 and explicit mode required')
    owner=None
    try:
        cls=serial_owner_class(config['source_root']) if mode=='serial' else owner_class(config['source_root'])
        owner=cls(config['policy'])
        rpc=owner.engine.llm_engine.collective_rpc
        def apply(operation):
            replies=rpc('jev_dual_measurement',args=(operation,))
            if len(replies)!=1:raise ValueError('TP1 worker required')
            return replies[0]
        if mode=='serial':
            installed=apply('serial_wire')
            if installed['policy_id']!=owner.bundle['policy_id']:raise ValueError('serial policy changed')
            attach_serial_score(owner)
        else:installed=owner.segmented_receipt
        counters=apply('install')
        commit_json(output/'startup.json',dict(runtime=owner.runtime,wire=installed,measurement=counters))
        result=measure(owner,rows,mode=mode,audio_token_id=config['audio_token_id'],
            telemetry=apply)
        result['input_sha256']=config['input_sha256'];validate_measurement(result)
        commit_json(output/'measurement.json',result)
        owner.close();owner=None;commit_json(output/'terminal.json',dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    finally:
        if owner is not None:owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('config','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=['serial','batched'],required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,a.mode)
