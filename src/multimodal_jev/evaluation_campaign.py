"""Reusable label-free inputs and one pool across fixed-policy evaluations.

Caller owns deployment, audited policy activation and scoring. This component
does not reuse predictions or policy-dependent KV across different adapters.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import threading
import time


class CampaignTimeline:
    def __init__(self):
        self.origin=time.monotonic(); self.events=[]; self.lock=threading.Lock()

    @contextmanager
    def stage(self, stage, job_id=None):
        begin=time.monotonic();status='failed'
        try:
            yield
            status='complete'
        finally:
            end=time.monotonic()
            with self.lock:
                self.events.append(dict(stage=stage,job_id=job_id,start=begin-self.origin,
                                        end=end-self.origin,seconds=end-begin,status=status))

    def report(self):
        with self.lock: events=copy.deepcopy(self.events)
        totals={}
        for event in events:totals[event['stage']]=totals.get(event['stage'],0)+event['seconds']
        return dict(wall_seconds=time.monotonic()-self.origin,stage_work_seconds=totals,events=events,
                    note='Overlapping stage work must not be summed into wall time')


def tensor_bytes(row):
    return sum(t.numel()*t.element_size() for t in [*row['inputs'].values(),row['positions']])


class ResidentEvaluationInputs:
    """Bounded, preloaded immutable input corpus, independent of model policy.

    Shared tensors are read-only by contract. The existing evaluator checks
    their content before/after inference; per-job copies would defeat reuse.
    """
    def __init__(self, population, encode, *, io_workers, max_bytes):
        if type(io_workers)is not int or io_workers<1 or type(max_bytes)is not int or max_bytes<1:
            raise ValueError('positive configured input workers and memory budget required')
        self.assets={};self.values={};self.bytes=0;self.loads=0;self.hits=0
        if not population or len({a['id'] for a in population})!=len(population):
            raise ValueError('complete unique input population required')
        def load(asset):
            if set(asset)!={'id','audio_sha256','request_sha256'}:
                raise ValueError('label-free asset identity required')
            row,receipt=encode(copy.deepcopy(asset))
            if (set(row)!={'inputs','positions'} or any(receipt.get(k)!=v for k,v in asset.items())
                    or any(t.device.type!='cpu' for t in [*row['inputs'].values(),row['positions']])):
                raise ValueError('bound CPU tensors required')
            return row,receipt
        # Bounded submission avoids the executor retaining a corpus of pending
        # futures/results beyond the configured tensor memory allowance.
        with ThreadPoolExecutor(max_workers=io_workers) as executor:
            for start in range(0,len(population),io_workers):
                chunk=population[start:start+io_workers]
                futures=[executor.submit(load,a) for a in chunk]
                for asset,future in zip(chunk,futures,strict=True):
                    row,receipt=future.result();size=tensor_bytes(row)
                    if self.bytes+size>max_bytes:
                        raise ValueError('resident input budget exceeded; configure sharded corpus')
                    self.assets[asset['id']]=dict(asset)
                    self.values[asset['id']]=(row,copy.deepcopy(receipt))
                    self.bytes+=size;self.loads+=1

    def __call__(self, asset):
        if self.assets.get(asset['id'])!=asset:raise ValueError('resident input identity changed')
        row,receipt=self.values[asset['id']];self.hits+=1
        return dict(inputs=dict(row['inputs']),positions=row['positions']),copy.deepcopy(receipt)


def run_campaign(pool, jobs, *, evaluate, audit, timeline, cpu_workers):
    """Quiescent switches, a shared pool, and overlapped independent CPU audits.

    jobs hold caller-validated snapshot/adapter, resident encode and evaluation
    kwargs. Unique macro namespaces prevent collisions within the shared pool.
    All audit futures are drained before campaign timing stops.
    """
    if type(cpu_workers)is not int or cpu_workers<1 or not jobs or len({j['job_id'] for j in jobs})!=len(jobs):
        raise ValueError('unique configured campaign jobs and CPU workers required')
    results=[];pending=[]
    def audit_one(job,result):
        with timeline.stage('cpu_audit',job['job_id']):return audit(job,result)
    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
        for job in jobs:
            for future in pending:
                if future.done():future.result()
            args=dict(job['evaluation']);policy=args['policy_id']
            if pool.bundle['policy_id']!=policy:
                with timeline.stage('policy_switch',job['job_id']):
                    pool.switch(snapshot=job['snapshot'],policy_id=policy,adapter=job['adapter'])
            with timeline.stage('evaluation',job['job_id']):
                result=evaluate(pool,job['population'],job['encode'],**args,
                                batch_namespace=job['job_id'],timeline=timeline)
            results.append(result);pending.append(executor.submit(audit_one,job,result))
        audits=[future.result() for future in pending]
    return dict(results=results,audits=audits,timing=timeline.report(),formal_admitted=False)
