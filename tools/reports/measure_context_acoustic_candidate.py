"""Matched service32 baseline/acoustic-batch timing and numerical evidence."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from contextlib import contextmanager

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.context_rollout_batch import bind_rollout
from tools.reports.context_batched_service_candidate import score_rows
from tools.reports.context_acoustic_owner_candidate import owner_class


@contextmanager
def fresh_audio_pass(owner, index):
    """Unique MM cache namespace per pass; tensors and policy IDs stay intact."""
    engine = owner.engine
    class FreshAudioView:
        def encode(self, prompts, **kwargs):
            fresh = []
            for prompt in prompts:
                uuids = prompt['multi_modal_uuids']
                if set(uuids) != {'audio'} or len(uuids['audio']) != 1:
                    raise ValueError('one explicit audio cache identity required')
                fresh.append(dict(prompt, multi_modal_uuids={
                    'audio': [f'acoustic-measurement-pass-{index}:'+uuids['audio'][0]]}))
            return engine.encode(fresh, **kwargs)
    owner.engine = FreshAudioView()
    try:
        yield
    finally:
        owner.engine = engine


def compare(reports,rows):
    if set(reports)!={'baseline','acoustic'}:raise ValueError('two matched arms required')
    first=reports['baseline'];anchor=first['passes'][0]['records']; errors=[];means={};calls={}
    sample_matches=True
    for mode,r in reports.items():
        if any(r[k]!=first[k] for k in ('policy_id','input_sha256','rows')) or r['rows']!=len(rows):
            raise ValueError('same policy and exact inputs required')
        if [p['kind'] for p in r['passes']]!=['warmup','timed','timed','reversed']:
            raise ValueError('four fixed passes required')
        calls[mode]=[]
        for index,p in enumerate(r['passes']):
            expected=list(range(len(rows)))
            if index==3:expected.reverse()
            if p['order']!=expected or len(p['records'])!=len(rows) or not math.isfinite(p['seconds']) or p['seconds']<=0:
                raise ValueError('complete finite measurement required')
            lens=[len(row['inputs']['input_ids'][0]) for row in rows]
            groups=p['telemetry']['counts']['language']
            if (Counter(n for g in groups for n in g)!=Counter(lens)
                    or any(len(g)>32 or sum(g)>131072 for g in groups)):
                raise ValueError('actual scheduled coverage/capacity differs')
            acoustic=p['acoustic_counts']
            if not acoustic['conv'] or not acoustic['encoder']:raise ValueError('actual acoustic telemetry required')
            memory=p['telemetry']
            if not 0<memory['peak_allocated_bytes']<=memory['peak_reserved_bytes']:
                raise ValueError('positive ordered allocator peaks required')
            if len(p['sampling'])!=len(rows):raise ValueError('complete sampling population required')
            for slot,(row,record) in enumerate(zip(rows,p['records'],strict=True)):
                validate_logits_record(record)
                sampled=bind_rollout(record,**row,policy_id=r['policy_id'],group_size=16,seed=317+slot)['receipt']
                if sampled!=p['sampling'][slot]:raise ValueError('input or sampling binding differs')
                ref=anchor[slot]
                if record['request_identity']!=ref['request_identity'] or record['request_id']!=ref['request_id']:
                    raise ValueError('request identity/order differs')
                errors.append((max(abs(a-b) for a,b in zip(record['logits'],ref['logits'],strict=True)),
                               max(abs(a-b) for a,b in zip(record['prob'],ref['prob'],strict=True))))
                sample_matches &= sampled==first['passes'][0]['sampling'][slot]
            calls[mode].append(dict(conv_calls=len(acoustic['conv']),
                conv_chunks=sum(s[0] for s in acoustic['conv']),encoder_calls=len(acoustic['encoder'])))
        means[mode]=sum(p['seconds'] for p in r['passes'][1:3])/2
    logit=max(e[0] for e in errors);prob=max(e[1] for e in errors)
    return dict(mean_seconds=means,speedup=means['baseline']/means['acoustic'],
        max_logit_error=logit,max_probability_error=prob,exact_probabilities_and_logits=logit==prob==0,
        exact_sampling=sample_matches,acoustic_calls=calls,training_dispatch_allowed=False,
        scope='fixed-policy service32 comparison; no training speed or gradient admission')


def run(config,output,mode):
    import torch
    from multimodal_jev.local_runtime_storage import require_training_storage
    from tools.reports.context_bulk_rpc_candidate import validate_rows
    output=Path(output);require_training_storage(output);output.mkdir(exist_ok=False)
    for path,h in config['pins'].items():
        if file_hash(path)!=h:raise ValueError('frozen dependency changed: '+path)
    if file_hash(config['inputs'])!=config['input_sha256']:raise ValueError('inputs changed')
    rows=validate_rows(torch.load(config['inputs'],map_location='cpu',weights_only=True))
    if len(rows)!=128 or mode not in ('baseline','acoustic'):raise ValueError('fixed TRAIN128 comparison required')
    owner=None
    try:
        owner=owner_class(config['source_root'],acoustic=mode=='acoustic')(config['policy'])
        rpc=owner.engine.llm_engine.collective_rpc
        def telemetry(method,op):
            values=rpc(method,args=(op,))
            if len(values)!=1:raise ValueError('TP1 worker required')
            return values[0]
        telemetry('jev_dual_measurement','install');telemetry('jev_acoustic_counters','install')
        commit_json(output/'startup.json',dict(runtime=owner.runtime,installation=owner.segmented_receipt))
        passes=[]
        for index in range(4):
            order=list(range(len(rows)))
            if index==3:order.reverse()
            telemetry('jev_dual_measurement','reset');telemetry('jev_acoustic_counters','reset')
            started=time.perf_counter()
            with fresh_audio_pass(owner,index):
                records,plan=score_rows(owner,[rows[i] for i in order],policy_id=owner.bundle['policy_id'],audio_token_id=config['audio_token_id'])
            elapsed=time.perf_counter()-started
            counts=telemetry('jev_dual_measurement','read');acoustics=telemetry('jev_acoustic_counters','read')
            canonical=[None]*len(rows);sampling=[None]*len(rows)
            for slot,record in zip(order,records,strict=True):
                canonical[slot]=record
                sampling[slot]=bind_rollout(record,**rows[slot],policy_id=owner.bundle['policy_id'],group_size=16,seed=317+slot)['receipt']
            value=dict(kind='warmup' if index==0 else 'reversed' if index==3 else 'timed',order=order,
                seconds=elapsed,records=canonical,sampling=sampling,plan=plan,telemetry=counts,acoustic_counts=acoustics)
            commit_json(output/f'pass-{index}.json',value);passes.append(value)
        result=dict(mode=mode,policy_id=owner.bundle['policy_id'],input_sha256=config['input_sha256'],rows=len(rows),passes=passes)
        commit_json(output/'measurement.json',result);owner.close();owner=None
        commit_json(output/'terminal.json',dict(exit_code=0));return result
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    finally:
        if owner is not None:owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['baseline','acoustic'],required=True);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output,a.mode)
