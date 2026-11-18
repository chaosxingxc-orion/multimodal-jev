"""Finite serial/batched model comparison; never selects policies or trains."""
import argparse
import copy
import json
from pathlib import Path
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_rollout_batch import bind_rollout
from tools.reports.context_batched_service_candidate import score_rows


def install_counters(model):
    """Run inside the actual worker, after startup and profile installation."""
    import torch
    if hasattr(model, '_jev_measurement_handles'):
        raise ValueError('fresh measurement hooks required')
    model._jev_measurement_counts = dict(language=[], audio=[])
    def language(module, args, kwargs):
        positions = kwargs.get('positions', args[0] if args else None)
        if positions is None or positions.ndim != 1:
            raise ValueError('flat full-prefill measurement positions required')
        values = positions.detach().cpu()
        starts = (values == 0).nonzero().flatten().tolist()
        if not starts or starts[0] != 0:
            raise ValueError('complete measured requests must start at zero')
        ends = starts[1:] + [len(values)]
        lengths = [b-a for a,b in zip(starts,ends)]
        for a,b in zip(starts,ends):
            if not torch.equal(values[a:b], torch.arange(b-a, dtype=values.dtype)):
                raise ValueError('invalid measured request positions')
        model._jev_measurement_counts['language'].append(lengths)
    def audio(module, args, kwargs):
        lens = kwargs.get('feature_lens', args[1] if len(args)>1 else None)
        if lens is None: raise ValueError('measured audio feature lengths missing')
        model._jev_measurement_counts['audio'].append(lens.detach().cpu().tolist())
    model._jev_measurement_handles = [
        model.language_model.model.layers[0].self_attn.register_forward_pre_hook(language, with_kwargs=True),
        model.audio_tower.register_forward_pre_hook(audio, with_kwargs=True)]
    return dict(installed=True, scope='first language attention per model forward and outer audio tower calls')


def reset_counters(model):
    import torch
    torch.npu.synchronize()
    model._jev_measurement_counts = dict(language=[], audio=[])
    torch.npu.reset_peak_memory_stats()
    return dict(reset=True)


def read_counters(model):
    import torch
    torch.npu.synchronize()
    return dict(counts=copy.deepcopy(model._jev_measurement_counts),
                peak_allocated_bytes=torch.npu.max_memory_allocated(),
                peak_reserved_bytes=torch.npu.max_memory_reserved(),
                scope='worker allocator only, not device total memory')


def measure(owner, rows, *, mode, audio_token_id, telemetry, clock=time.perf_counter):
    if mode not in ('serial', 'batched') or not rows:
        raise ValueError('explicit finite comparison mode and rows required')
    policy = owner.bundle['policy_id']; passes=[]
    orders = [list(range(len(rows)))]*3 + [list(reversed(range(len(rows))))]
    for index, order in enumerate(orders):
        telemetry('reset')
        started=clock()
        if mode=='serial':
            records=[owner.score(**rows[i],audio_token_id=audio_token_id) for i in order]
            plan=dict(encode_calls=len(rows))
        else:
            records,plan=score_rows(owner,[rows[i] for i in order],policy_id=policy,audio_token_id=audio_token_id)
        elapsed=clock()-started
        counters=telemetry('read')
        if not 0<elapsed<float('inf') or owner.bundle['policy_id']!=policy or len(records)!=len(rows):
            raise ValueError('finite elapsed time and complete unchanged policy required')
        canonical=[None]*len(rows);sampling=[None]*len(rows)
        for slot,record in zip(order,records,strict=True):
            canonical[slot]=record
            sampling[slot]=bind_rollout(record,**rows[slot],policy_id=policy,group_size=16,seed=317+slot)['receipt']
        if index and (canonical!=passes[0]['records'] or sampling!=passes[0]['sampling']):
            raise ValueError('repeat or request-neighbor/order invariance failed')
        passes.append(dict(kind=('warmup' if index==0 else 'reversed' if index==3 else 'timed'),
            order=order,seconds=elapsed,records=canonical,sampling=sampling,plan=plan,telemetry=counters))
    return dict(mode=mode,policy_id=policy,rows=len(rows),passes=passes,
                scope='finite fixed-policy requests with synchronized profiling hooks; not actor throughput')


def compare(serial, batched):
    if serial['mode']!='serial' or batched['mode']!='batched':
        raise ValueError('ordered serial/batched pair required')
    if any(serial[k]!=batched[k] for k in ('policy_id','rows','input_sha256')):
        raise ValueError('different policy or inputs')
    for report in (serial,batched):
        if [p['kind'] for p in report['passes']]!=['warmup','timed','timed','reversed']:
            raise ValueError('complete predefined passes required')
    for a,b in zip(serial['passes'],batched['passes'],strict=True):
        if a['records']!=b['records'] or a['sampling']!=b['sampling']:
            raise ValueError('serial/batched probabilities or G16 sampling differ')
    calls=[];means=[]
    for report in (serial,batched):
        timed=report['passes'][1:3]
        means.append(sum(p['seconds'] for p in timed)/2)
        counts=[]
        for p in timed:
            groups=p['telemetry']['counts']['language']
            if not groups or sum(map(len,groups))!=report['rows']:
                raise ValueError('actual language forward coverage differs')
            counts.append(len(groups))
        calls.append(counts)
    return dict(exact_probabilities_and_G16=True,rows=serial['rows'],
                serial_seconds=means[0],batched_seconds=means[1],time_ratio=means[0]/means[1],
                actual_language_calls=calls,actual_call_reduction=all(b<a for a,b in zip(*calls)),
                scope='two timed passes with measurement hooks; startup/switch/actor/checkpoint excluded')


def run(config, output, mode):
    import torch
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json
    from tools.reports.context_bulk_rpc_candidate import validate_rows
    from tools.reports.context_batched_owner_candidate import owner_class
    from context_switch_rollout import SwitchingContextRollout
    output=Path(output);require_training_storage(output);output.mkdir(exist_ok=False)
    for name,digest in config['pins'].items():
        if file_hash(name)!=digest:raise ValueError('measurement dependency changed: '+name)
    if file_hash(config['inputs'])!=config['input_sha256']:
        raise ValueError('actual fixed input payload differs')
    rows=validate_rows(torch.load(config['inputs'],map_location='cpu',weights_only=True))
    if len(rows)!=128 or mode not in ('serial','batched'):
        raise ValueError('fixed real TRAIN128 and explicit mode required')
    owner=None
    try:
        cls=SwitchingContextRollout if mode=='serial' else owner_class(config['source_root'])
        owner=cls(config['policy'])
        rpc=owner.engine.llm_engine.collective_rpc
        def apply(fn):
            result=rpc('apply_model',args=(fn,))
            if len(result)!=1:raise ValueError('TP1 single-worker measurement required')
            return result[0]
        installed=apply(install_counters)
        commit_json(output/'startup.json',dict(runtime=owner.runtime,measurement=installed,
            candidate=getattr(owner,'segmented_receipt',None)))
        result=measure(owner,rows,mode=mode,audio_token_id=config['audio_token_id'],
            telemetry=lambda operation:apply(reset_counters if operation=='reset' else read_counters))
        result['input_sha256']=config['input_sha256']
        commit_json(output/'measurement.json',result)
        owner.close();owner=None
        commit_json(output/'terminal.json',dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    finally:
        if owner is not None:owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['serial','batched'],required=True);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output,a.mode)
