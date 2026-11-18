"""After current parity queue: acoustic probe, bucket100, independent comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_macro_training import wait_driver
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports import context_acoustic_probe_operations as acoustic
from tools.reports import context_bucket_training_operations as bucket
from tools.reports.audit_context_macro_training import audit_collected
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.context_macro_efficiency_summary import summarize


def read(p):return json.loads(Path(p).read_text())


def pins(c):
    for p,h in c['pins'].items():
        if file_hash(p)!=h:raise ValueError('efficiency dependency changed: '+p)


def compare_training(baseline,candidate,baseline_audit,candidate_audit):
    if any(r['status']!='finite_dp2_efficiency_audited' for r in (baseline_audit,candidate_audit)):
        raise ValueError('complete independent training audits required')
    if baseline_audit['image']!=candidate_audit['image']:raise ValueError('same image required')
    for ops,audit in ((baseline,baseline_audit),(candidate,candidate_audit)):
        if ops['manifest_sha256']!=audit['files_sha256']:raise ValueError('audit package differs')
    stats=[];orders=[];padding=[]
    from multimodal_jev.context_length_buckets import plan
    for ops in (baseline,candidate):
        root=Path(ops['operations_root'])/'collected/run'
        stats.append(summarize(root,start_update=0,stop_update=100,warmup_updates=2))
        orders.append([read(root/f'batch-{i:04d}.json')['records'] for i in range(100)])
    if orders[0]!=orders[1] or [s['reference_misses'] for s in stats[0]['steps']]!=[s['reference_misses'] for s in stats[1]['steps']]:
        raise ValueError('actual membership/order/reference cache differs')
    root=Path(candidate['operations_root'])/'collected/run'
    for i in range(100):
        records=[r['behavior_record'] for r in read(root/f'rollout-{i:04d}.json')['queries']]
        lengths=[(r['request_identity']['inputs']['input_ids']['shape'][-1],
                  r['request_identity']['inputs']['input_features']['shape'][-1]) for r in records]
        _,report=plan(lengths,microbatch=8,world_size=2,**candidate['length_buckets']);padding.append(report)
    return dict(baseline=stats[0],candidate=stats[1],padding=padding,
        all_updates_speedup=stats[0]['all_updates']['complete_seconds']/stats[1]['all_updates']['complete_seconds'],
        after_warmup_speedup=stats[0]['measured']['complete_seconds']/stats[1]['measured']['complete_seconds'],
        saved_seconds=stats[0]['all_updates']['complete_seconds']-stats[1]['all_updates']['complete_seconds'],
        source_audits=[baseline_audit,candidate_audit],scope='bucket-only matched100; no efficacy claim')


def run(c,provenance):
    pins(c);a=read(c['acoustic_operations']);b=read(c['bucket_operations'])
    acoustic.validate(a);bucket.validate(b)
    root=Path(c['output']);check_paths({'successor':root});root.mkdir(parents=True,exist_ok=False)
    write_new(root/'config.json',c);write_new(root/'provenance.json',provenance)
    proc=Path('/proc')/str(os.getpid())
    write_new(root/'driver.json',dict(pid=os.getpid(),start_ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19],
        cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
    try:
        predecessor=read(c['predecessor_driver'])
        write_new(root/'waiting.json',dict(process=predecessor,terminal=c['predecessor_terminal']))
        wait_driver(predecessor,Path(c['predecessor_terminal']));pins(c)
        t=read(c['predecessor_terminal'])
        if t.get('exit_code')!=0 or t.get('status')!='three_target_width_parity_audited':
            raise ValueError('existing parity queue must complete successfully')
        baseline=b['parent_operations'];base_audit=audit_collected(baseline)
        write_new(root/'baseline-audit.json',base_audit)
        result=acoustic.namespace(a)['launch'](a,provenance)
        write_new(root/'acoustic-comparison.json',result)
        # Numerical deviation is retained and blocks acoustic promotion; the
        # bucket-only arm continues on unchanged baseline inference weights/code.
        pins(c);bucket.run(b,provenance)
        result=collect_and_audit(b,validator=bucket.validate,command_builder=bucket.command,
            output_auditor=bucket.audit_outputs,result_dirs=('run','journal','service-0','service-1'))
        write_new(root/'bucket-audit.json',result)
        comparison=compare_training(baseline,b,base_audit,result)
        write_new(root/'bucket-comparison.json',comparison)
        write_new(root/'terminal.json',dict(exit_code=0,status='acoustic_and_bucket_efficiency_compared',
            formal_dispatched=False,automatic_promotion=False))
    except BaseException as exc:
        write_new(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    import torch
    torch.set_num_threads(1);run(c,provenance)
