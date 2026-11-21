"""Versioned four-card selection curve; NPU work and CPU audits overlap.

Sealed checkpoint identity admits inference while full training audit runs.
Every output audit uses the existing parity/score/transport implementation.
Final publication still requires the independent full training verdict.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType,SimpleNamespace
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import verify
from tools.reports import prepare_committed_selection as preparation
from tools.reports import context_composed_pipeline as lifecycle
from tools.reports import context_train_parity_operations as parity
from tools.reports import multidevice_selection_operations as evaluation
from tools.reports import audit_context_train_parity as transport
from tools.reports.context_async_pipeline import ReadyQueue
from tools.reports.context_completed_stage_receipts import check_tree
from tools.reports.prepare_objective_training_candidate import read_bound


def validate(ops):
    preparation.audit(ops['selection_package'],{})
    fn=FunctionType(parity.validate.__code__,dict(parity.validate.__globals__,audit_package=lambda *a:None,command=lifecycle.command))
    result=fn(dict(ops,parity_package=ops['selection_package']))
    from tools.reports.context_release_deployment import audit
    audit(lifecycle.command(ops,result[0],result[1],producer=True),result[1]);return result


def audit_outputs(bound,p):
    fn=FunctionType(transport.audit.__code__,dict(transport.audit.__globals__,audit_package=preparation.audit,
        audit_outputs=evaluation.audit_outputs,command=lifecycle.command))
    proof=fn(dict(bound,parity_package=bound['selection_package']),p)
    proof.update(status='committed_grpo_selection_inference_audited_training_audit_pending',training_audited=False,
                 final_publication_allowed=False)
    return proof


def check(c):
    from multimodal_jev.evaluation_performance_gate import require_performance_admission
    require_performance_admission(c)
    if c['steps']!=[634,64,128,192,256,320,384,448,512,576] or c['monitor_interval_seconds']!=900:
        raise ValueError('endpoint-first complete registered curve required')
    if c.get('dispatch_allowed') is not True:raise ValueError('explicit inference dispatch required')
    snapshot=c['source_snapshot']
    if Path(__file__).resolve().parents[2]!=Path(snapshot['root']):raise ValueError('frozen controller required')
    verify(snapshot['root'],manifest_sha256=snapshot['manifest_sha256'],io_workers=4)
    for pin in c['pins']:read_bound(pin)
    cpu=json.loads(read_bound(c['cpu_report']))
    if cpu['counts']!=dict(tests=c['cpu_tests'],failures=0,errors=0,skipped=0) or cpu['state']['ExitCode'] or cpu['state']['OOMKilled']:
        raise ValueError('fixed image tests required')
    tested=json.loads((Path(c['cpu_source'])/'FILES.json').read_text())
    for pin in c['pins']:
        rel=str(Path(pin['path']).relative_to(snapshot['root']))
        if tested.get(rel)!=pin['sha256']:raise ValueError('tested controller differs')


def run(c,p):
    check(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False);commit_json(root/'config.json',c);commit_json(root/'provenance.json',p)
    queue=ReadyQueue(root/'events');futures=[]
    try:
        with ThreadPoolExecutor(max_workers=c['cpu_audit_workers']) as pool:
            for step in c['steps']:
                for future in futures:
                    if future.done():future.result()
                stage=c['locations'][str(step)];spec=dict(c['package_template'],step=step,output=stage['package'],remote_root=stage['remote_root'])
                spec['manifest_sha256']=preparation.prepare(spec,p)
                ops=dict(c['deployment'],selection_package=spec,package=spec['output'],manifest_sha256=spec['manifest_sha256'],
                         operations_root=stage['operations_root'],tag=stage['tag'],finite_training_dispatch_allowed=True)
                work=root/f'point-{step:04d}';work.mkdir();commit_json(work/'deployment.json',ops)
                def locked(paths,ops=ops):
                    lifecycle.EFFECTS.check_paths(paths)
                    package=Path(ops['package']);manifest=json.loads((package/'PREPARED.json').read_text())
                    check_tree(package,dict(manifest['files'],**{'PREPARED.json':ops['manifest_sha256']}))
                effects=SimpleNamespace(**dict(vars(lifecycle.EFFECTS),check_paths=locked))
                from tools.reports.context_cached_formal_successor import monitored_training
                launch=lambda current,provenance:lifecycle._launch(current,provenance,validate=validate,queue=queue,priority=0,producer=True,effects=effects)
                monitor=lifecycle.bind(monitored_training,formal=SimpleNamespace(run=launch))
                result=monitor(ops,p,work)
                if result['exit_code']:raise ValueError('selection failed; exact collection retained')
                stem=Path(ops['operations_root'])/ops['tag']
                bound=dict(selection_package=spec,deployment=ops,**{k:str(stem)+suffix for k,suffix in (
                    ('launch','-launch.json'),('inspect','-inspect.json'),('wait','-wait.json'),('transport','-collected.transport.json'),
                    ('collection','-collected'),('archive','-collected.tar'))})
                commit_json(work/'audit-config.json',bound)
                def audit_one(bound=bound,work=work):
                    proof=audit_outputs(bound,p);commit_json(work/'audit.json',proof);return proof
                futures.append(pool.submit(audit_one))
            results=[f.result() for f in futures]
        commit_json(root/'results.json',results)
        commit_json(root/'terminal.json',dict(exit_code=0,status='grpo_selection_curve_inference_audited_training_audit_pending',
                    points=len(results),final_publication_allowed=False,transfer_pending=True))
    except BaseException as e:
        queue.abort();commit_json(root/'terminal.json',dict(exit_code=1,error=repr(e),restart=False));raise


def detach(template,local):
    c,_=load_config(template,local);check(c);root=Path(c['output']+'-driver');root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.run_committed_selection','--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,argv=argv,started_unix=time.time(),process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19]);commit_json(root/'started.json',receipt);return receipt


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--local',required=True);parser.add_argument('--action',choices=['run','detach'],required=True);a=parser.parse_args()
    c,p=load_config(a.config,a.local);print(json.dumps(detach(a.config,a.local) if a.action=='detach' else run(c,p)))
