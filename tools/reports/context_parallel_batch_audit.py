"""Process-parallel original InfoNCE target audit with measured equivalence.

Standalone CPU evidence component, not a replacement for checkpoint/transport
or NPU acceptance. Worker processes call the unchanged original check_batch.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.parallel_evidence import inventory, workers

_STATE = None


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',', ':')).encode()).hexdigest()


def initialize(population, task, policy, width, world):
    global _STATE
    _STATE = (population, task, policy, width, world)
    import torch
    torch.set_num_threads(1)


def check_one(item):
    from tools.reports.audit_context_infonce_training import check_batch
    index, path, expected_hash, planned = item
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('batch bytes changed: '+path)
    batch = json.loads(raw)
    population, task, policy, width, world = _STATE
    real = check_batch(batch, planned, population, task,
        policy=policy+':instruction-update'+str(index),width=width,world=world)
    return dict(index=index,real_queries=real,batch_sha256=expected_hash,batch_semantic_sha256=identity(batch))


def audit_batches(items, population, task, *, policy, width, world, cpu_workers=4):
    workers(cpu_workers)
    indices = [v[0] for v in items]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError('unique ordered batches required')
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(population,task,policy,width,world)) as pool:
        results = list(pool.map(check_one, items, chunksize=1))
    if [r['index'] for r in results] != indices:
        raise ValueError('parallel audit lost or reordered batches')
    return results,time.perf_counter()-start


def read_bound(binding):
    path=Path(binding['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=binding['sha256']:
        raise ValueError('audit input changed: '+str(path))
    return json.loads(raw)


def run(c, provenance):
    from multimodal_jev.context_replica_schedule import load_schedule
    from multimodal_jev.context_topology_batch_plan import derive
    from multimodal_jev.slurp_portable_release import load
    from tools.reports.audit_context_infonce_training import check_artifact_inventory
    workers(c['cpu_workers']);workers(c['io_workers'])
    if type(c['serial_comparison_batches']) is not int or not 1 <= c['serial_comparison_batches'] <= 32:
        raise ValueError('explicit bounded differential panel required')
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    commit_json(out/'config.json',c);commit_json(out/'provenance.json',provenance)
    started=time.perf_counter()
    try:
        for name,digest in c['pins'].items():
            if file_hash(name)!=digest:raise ValueError('auditor code changed: '+name)
        cfg=read_bound(c['training_config']);transport=read_bound(c['transport'])
        plan=derive(cfg['batch_config'])
        if cfg['experiment_kind']!='formal_infonce_source':raise ValueError('full formal source required')
        package=Path(c['package']);collection=Path(c['collection']);run_root=collection/'run'
        if c['training_config']['path']!=str(package/'config.json'):
            raise ValueError('actual package config required')
        begin=time.perf_counter();actual=inventory(collection,io_workers=c['io_workers'])
        if actual!=transport['files']:raise ValueError('collection differs from original transport')
        inventory_seconds=time.perf_counter()-begin
        rows,task,_=load(cfg['release'],cfg['release_sha256'],'learning',purpose='optimize')
        population={r['id']:r for r in rows}
        batches,_=load_schedule(package/'source/schedule',cfg['schedule_sha256'],
            release_sha256=cfg['release_sha256'],population=population,batch_config=cfg['batch_config'],
            parent_root=package/'source/parent-schedule',parent_manifest_sha256=cfg['parent_schedule_sha256'],
            parent_release_sha256=cfg['schedule_release_sha256'])
        check_artifact_inventory(run_root,cfg,plan['dp_replicas'])
        if cfg['start_update']!=0 or cfg['stop_update']!=len(batches):raise ValueError('complete window required')
        items=[(i,str(run_root/f'batch-{i:04d}.json'),actual[f'run/batch-{i:04d}.json'],b)
               for i,b in enumerate(batches)]
        kwargs=dict(policy=cfg['source_policy_id'],width=plan['global_batch'],world=plan['dp_replicas'])
        # Same exact original checker, same actual inputs, equal result objects.
        n=min(c['serial_comparison_batches'],len(items))
        initialize(population,task,kwargs['policy'],kwargs['width'],kwargs['world'])
        begin=time.perf_counter();serial=[check_one(v) for v in items[:n]];serial_seconds=time.perf_counter()-begin
        panel,parallel_panel_seconds=audit_batches(items[:n],population,task,**kwargs,cpu_workers=c['cpu_workers'])
        if panel!=serial:raise ValueError('serial/parallel differential mismatch')
        results,parallel_seconds=audit_batches(items,population,task,**kwargs,cpu_workers=c['cpu_workers'])
        if sum(v['real_queries'] for v in results)!=len(population)*cfg['batch_config']['epochs']:
            raise ValueError('full real exposure count differs')
        commit_json(out/'batches.json',results)
        report=dict(status='parallel_source_batches_verified',batches=len(results),
            real_exposures=sum(v['real_queries'] for v in results),cpu_workers=c['cpu_workers'],io_workers=c['io_workers'],
            inventory_seconds=inventory_seconds,serial_panel_seconds=serial_seconds,
            parallel_panel_seconds=parallel_panel_seconds,comparison_batches=n,serial_parallel_equal=True,
            parallel_full_batch_seconds=parallel_seconds,total_seconds=time.perf_counter()-started,
            model_calls=0,training_dispatch_allowed=False,full_training_audit=False,
            scope='Full source batch targets/order/weights/rank permutation; checkpoint and NPU audit remain separate')
        commit_json(out/'report.json',report);commit_json(out/'terminal.json',dict(exit_code=0,status=report['status']))
        return report
    except BaseException as exc:
        commit_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance),indent=2))
