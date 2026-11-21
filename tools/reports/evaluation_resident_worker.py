"""One resident engine and immutable input shard across a policy campaign.

Numerical model/runtime is imported from a separately frozen, authenticated
source. This worker changes orchestration only. No TEST gold or training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import time

import torch

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.span_request_identity import tensor_identity
_spec=importlib.util.spec_from_file_location('campaign_timing',Path(__file__).parents[2]/'src/multimodal_jev/evaluation_campaign.py')
_timing=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_timing)
CampaignTimeline,tensor_bytes=_timing.CampaignTimeline,_timing.tensor_bytes


def load_input(entry, cache):
    raw=(Path(cache)/(entry['cache_key']+'.pt')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['cache_sha256']:
        raise ValueError('input cache bytes changed')
    value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    row=dict(inputs=value['inputs'],positions=value['positions'])
    actual={k:tensor_identity(v) for k,v in dict(row['inputs'],positions=row['positions']).items()}
    if actual!=entry['tensors']:raise ValueError('input tensors differ from independent audit')
    return row


def batches(entries, width, token_budget):
    order=sorted(range(len(entries)),key=lambda i:(entries[i]['tensors']['input_ids']['shape'][-1],entries[i]['id']))
    current=[];tokens=0
    for i in order:
        n=entries[i]['tensors']['input_ids']['shape'][-1]
        if n>token_budget:raise ValueError('one request exceeds batch tokens')
        if current and (len(current)==width or tokens+n>token_budget):
            yield current;current=[];tokens=0
        current.append(i);tokens+=n
    if current:yield current


def capacity_policy(original, capacity):
    """Tune sequence/token/KV budgets together in a separately frozen run."""
    policy=dict(original)
    if capacity is None:return policy
    if (set(capacity)!={'batch_queries','max_batch_tokens','kv_cache_memory_bytes'}
            or type(capacity['batch_queries'])is not int or capacity['batch_queries'] not in (32,48,64,128)
            or type(capacity['max_batch_tokens'])is not int
            or not policy['max_model_len']<=capacity['max_batch_tokens']<=capacity['batch_queries']*policy['max_model_len']
            or type(capacity['kv_cache_memory_bytes'])is not int
            or not 1024**3<=capacity['kv_cache_memory_bytes']<=48*1024**3):
        raise ValueError('bounded coupled sequence/token/KV capacity required')
    policy.update(max_batch_queries=capacity['batch_queries'],max_batch_tokens=capacity['max_batch_tokens'],
                  kv_cache_memory_bytes=capacity['kv_cache_memory_bytes'])
    return policy


def submission_capacity(policy, queries):
    """Allow a larger submission queue while retaining bounded scheduler KV.

    Changing the owner's API budget does not change vLLM's max_num_seqs or
    max_num_batched_tokens. The native scheduler processes complete requests in
    waves; chunked prefill remains disabled.
    """
    if type(queries)is not int or queries not in (32,48,64,128,256):
        raise ValueError('explicit submission width through 256 required')
    return dict(max_batch_queries=queries,max_batch_tokens=queries*policy['max_model_len'])


def compare_records(actual, expected, policy):
    if (len(actual)!=len(expected) or len({r['id'] for r in actual})!=len(actual)
            or len({r['id'] for r in expected})!=len(expected)
            or {r['id'] for r in actual}!={r['id'] for r in expected}):
        raise ValueError('complete identical prediction population required')
    prior={r['id']:r for r in expected};prob=logit=0.;flips=top=0
    for r in actual:
        old=prior[r['id']]
        if r['policy_id']!=policy or old['policy_id']!=policy or r['request_identity']!=old['request_identity']:
            raise ValueError('policy/input identity changed')
        p,q=r['prob'],old['prob'];a,b=r['logits'],old['logits']
        import math
        if not p or any(not math.isfinite(v) for v in [*p,*q,*a,*b]):
            raise ValueError('complete finite candidate outputs required')
        prob=max(prob,max(abs(x-y) for x,y in zip(p,q,strict=True)))
        logit=max(logit,max(abs(x-y) for x,y in zip(a,b,strict=True)))
        flips+=sum((x>=.5)!=(y>=.5) for x,y in zip(p,q,strict=True))
        top+=max(range(len(p)),key=p.__getitem__)!=max(range(len(q)),key=q.__getitem__)
    return dict(recordings=len(actual),probability_max_abs=prob,logit_max_abs=logit,
                threshold_flips=flips,top1_changes=top,exact=prob==0 and logit==0)


def run(c, rank):
    # Do not let the candidate workspace shadow the accepted numerical runtime.
    from tools.reports.context_bounded_rollout_candidate import owner_class
    from tools.reports.context_dual_runtime_candidate import verify_runtime
    from tools.jev.context_cache_receipts import parse_stream
    root=Path(c['output'])/f'worker-{rank}';root.mkdir(exist_ok=False)
    torch.set_num_threads(c['torch_threads'])
    timeline=CampaignTimeline();engine=None
    try:
        with timeline.stage('input_manifest'):
            service=json.loads(Path(c['service_configs'][rank]).read_text())
            verify_runtime(service)
            stream=Path(c['cache_receipts']['path']).read_bytes()
            if hashlib.sha256(stream).hexdigest()!=c['cache_receipts']['sha256']:
                raise ValueError('cache receipt stream changed')
            entries=[r['value'] for r in parse_stream(stream.decode())[0][1:-1]]
            if len(entries)!=c['expected_population']:raise ValueError('full population differs')
            if c['limit'] is not None:entries=entries[:c['limit']]
            # Four disjoint static shards, longest-first balanced by acoustic +
            # text cost. Membership is input-only and stable across policies.
            shards=[[] for _ in c['devices']];loads=[0]*len(shards)
            for entry in sorted(entries,key=lambda e:-(e['tensors']['input_ids']['shape'][-1]+e['tensors']['input_features']['shape'][-1])):
                slot=min(range(len(loads)),key=lambda i:(loads[i],i))
                shards[slot].append(entry);loads[slot]+=entry['tensors']['input_ids']['shape'][-1]+entry['tensors']['input_features']['shape'][-1]
            entries=shards[rank]
        with timeline.stage('input_load'):
            rows=[];resident_bytes=0
            with ThreadPoolExecutor(max_workers=c['io_workers']) as pool:
                for start in range(0,len(entries),c['io_workers']):
                    futures=[pool.submit(load_input,e,c['cache']) for e in entries[start:start+c['io_workers']]]
                    for future in futures:
                        row=future.result();resident_bytes+=tensor_bytes(row)
                        if resident_bytes>c['resident_bytes_per_worker']:raise ValueError('resident input memory budget exceeded')
                        rows.append(row)
        policy=capacity_policy(service['policy'],c.get('capacity'))
        with timeline.stage('engine_start'):
            if c.get('kv_cache_memory_bytes') is not None:
                if c.get('capacity') is not None:raise ValueError('use one coupled capacity setting')
                budget=c['kv_cache_memory_bytes']
                if type(budget)is not int or not 1024**3<=budget<=32*1024**3:
                    raise ValueError('bounded explicit evaluation KV capacity required')
                policy['kv_cache_memory_bytes']=budget
            engine=owner_class(service['source_root'],scope_module=policy['device_scope_module'])(policy)
        commit_json(root/'ready.json',dict(pid=os.getpid(),recordings=len(rows),resident_bytes=resident_bytes,
                                         input_loads=len(rows),model_initializations=1,
                                         kv_cache_memory_bytes=policy['kv_cache_memory_bytes']))
        pending=[];summaries=[]
        def audit_point(point,records,jobroot):
            with timeline.stage('cpu_audit',point['job_id']):
                refraw=Path(point['predictions']['path']).read_bytes()
                if hashlib.sha256(refraw).hexdigest()!=point['predictions']['sha256']:
                    raise ValueError('frozen prediction evidence changed')
                ids={e['id'] for e in entries}
                expected=[r for r in json.loads(refraw)['records'] if r['id'] in ids]
                proof=compare_records(records,expected,point['policy_id'])
                commit_json(jobroot/'parity.json',proof)
                if not proof['exact']:raise ValueError('campaign outputs differ from same-policy frozen runtime')
                return proof
        with ThreadPoolExecutor(max_workers=1) as auditor:
            for point in c['points']:
                for future in pending:
                    if future.done():future.result()
                jobroot=root/point['job_id'];jobroot.mkdir()
                submitted=submission_capacity(policy,point.get('submission_queries',policy['max_batch_queries']))
                engine.max_batch_queries=submitted['max_batch_queries']
                engine.max_batch_tokens=submitted['max_batch_tokens']
                jobs=list(batches(entries,engine.max_batch_queries,engine.max_batch_tokens))
                commit_json(jobroot/'capacity.json',dict(submitted,
                    scheduler_max_sequences=policy['max_batch_queries'],scheduler_max_tokens=policy['max_batch_tokens'],
                    kv_cache_memory_bytes=policy['kv_cache_memory_bytes']))
                switch=None
                if engine.bundle['policy_id']!=point['policy_id']:
                    with timeline.stage('policy_switch',point['job_id']):
                        switch=engine.switch(snapshot=point['snapshot'],policy_id=point['policy_id'],adapter=point['adapter'])
                    commit_json(jobroot/'switch.json',switch)
                records=[]
                telemetry=[]
                measured=c.get('measure_first_batch',False)
                if measured:
                    if point==c['points'][0]:
                        engine.engine.llm_engine.collective_rpc('jev_dual_measurement',args=('install',))
                    engine.engine.llm_engine.collective_rpc('jev_dual_measurement',args=('reset',))
                for batch,indices in enumerate(jobs):
                    with timeline.stage('inference',point['job_id']):
                        values=engine.score_many([rows[i] for i in indices],audio_token_id=c['audio_token_id'])
                    if len(values)!=len(indices):raise ValueError('incomplete engine output')
                    for i,value in zip(indices,values,strict=True):records.append(dict(value,id=entries[i]['id']))
                    if batch%16==0:
                        from multimodal_jev.baseline import atomic_json
                        atomic_json(root/'progress.json',dict(job=point['job_id'],completed=len(records),total=len(rows)))
                if measured:
                    telemetry=engine.engine.llm_engine.collective_rpc('jev_dual_measurement',args=('read',))
                    commit_json(jobroot/'forward-telemetry.json',telemetry)
                with timeline.stage('output_write',point['job_id']):
                    commit_json(jobroot/'predictions.json',dict(records=records))
                pending.append(auditor.submit(audit_point,point,records,jobroot))
                summaries.append(dict(job_id=point['job_id'],recordings=len(records),policy_id=point['policy_id']))
            proofs=[future.result() for future in pending]
        with timeline.stage('engine_shutdown'):engine.close();engine=None
        result=dict(points=summaries,parity=proofs,timing=timeline.report(),input_loads=len(rows),
                    resident_bytes=resident_bytes,model_initializations=1,formal_admitted=False)
        commit_json(root/'result.json',result);commit_json(root/'terminal.json',dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),timing=timeline.report()));raise
    finally:
        if engine is not None:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--rank',type=int,required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.rank)
