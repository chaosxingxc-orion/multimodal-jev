"""Exact KV-queue wait, acoustic admission, matched100 training and full audit."""
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
from tools.reports import context_acoustic_training_operations as training
from tools.reports import context_acoustic_probe_operations as acoustic
from tools.reports.audit_context_macro_training import audit_collected
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.context_macro_efficiency_summary import summarize


def read(p): return json.loads(Path(p).read_text())


def pins(c):
    if not {c['operations'], c['probe_operations'], c['predecessor_driver']} <= set(c['pins']):
        raise ValueError('exact candidate, admission and predecessor must be pinned')
    for p, h in c['pins'].items():
        if file_hash(p) != h: raise ValueError('acoustic training dependency changed: '+p)


def compare(baseline, candidate, audits):
    training.validate(candidate)
    if candidate['parent_operations'] != baseline:
        raise ValueError('exact acoustic-only parent required')
    for ops, audit in zip((baseline, candidate), audits, strict=True):
        if (audit['status'] != 'finite_dp2_efficiency_audited'
                or audit['files_sha256'] != ops['manifest_sha256']):
            raise ValueError('complete independently audited100 required')
    if audits[0]['image'] != audits[1]['image']: raise ValueError('same image required')
    stats=[]; orders=[]
    for ops in (baseline, candidate):
        run=Path(ops['operations_root'])/'collected/run'
        stats.append(summarize(run,start_update=0,stop_update=100,warmup_updates=2))
        orders.append([read(run/f'batch-{i:04d}.json')['records'] for i in range(100)])
    if (orders[0] != orders[1] or [s['reference_misses'] for s in stats[0]['steps']]
            != [s['reference_misses'] for s in stats[1]['steps']]):
        raise ValueError('actual membership/order or reference cache differs')
    return dict(schema='jev-acoustic-training-efficiency-v1',baseline=stats[0],candidate=stats[1],
        all_updates_speedup=stats[0]['all_updates']['complete_seconds']/stats[1]['all_updates']['complete_seconds'],
        after_warmup_speedup=stats[0]['measured']['complete_seconds']/stats[1]['measured']['complete_seconds'],
        saved_seconds=stats[0]['all_updates']['complete_seconds']-stats[1]['all_updates']['complete_seconds'],
        source_audits=audits,matched_actual_queries=True,matched_reference_misses=True,
        scope='acoustic-only matched100 complete updates including activation; no efficacy/transfer claim')


def run(c, provenance):
    pins(c); ops=read(c['operations']); probe=read(c['probe_operations'])
    training.validate(ops); acoustic.validate(probe)
    root=Path(c['output']); check_paths({'acoustic_training_study':root})
    root.mkdir(parents=True,exist_ok=False)
    write_new(root/'config.json',c); write_new(root/'provenance.json',provenance)
    proc=Path('/proc')/str(os.getpid())
    write_new(root/'driver.json',dict(pid=os.getpid(),
        start_ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19],
        cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
    try:
        driver=read(c['predecessor_driver'])
        write_new(root/'waiting.json',dict(predecessor=driver,mechanism='exact_pidfd',remote_polling=False))
        wait_driver(driver,Path(c['predecessor_terminal'])); pins(c)
        terminal=read(c['predecessor_terminal'])
        if terminal.get('exit_code') != 0 or terminal.get('status') != 'kv_capacity_independently_compared':
            raise ValueError('successful terminal KV queue required; no implicit restart')
        # The KV candidate is not adopted here: this arm keeps the original 1GiB.
        base=Path(probe['operations_root']); tag=probe['tag']
        admission=acoustic.audit(probe,base/(tag+'-collected'),base/(tag+'-launch.json'))
        if not admission['exact_probabilities_and_logits'] or not admission['exact_sampling']:
            raise ValueError('exact acoustic numerical admission required')
        write_new(root/'acoustic-admission.json',admission)
        baseline=ops['parent_operations']; old=audit_collected(baseline)
        write_new(root/'baseline-audit.json',old)
        pins(c); released=dict(ops,finite_training_dispatch_allowed=True)
        write_new(root/'released-operations.json',released)
        training.run(released,provenance)
        new=collect_and_audit(released,validator=training.validate,command_builder=training.command,
            output_auditor=training.audit_outputs,result_dirs=('run','journal','service-0','service-1'))
        write_new(root/'candidate-audit.json',new)
        result=compare(baseline,released,[old,new]); write_new(root/'comparison.json',result)
        write_new(root/'terminal.json',dict(exit_code=0,status='acoustic_training100_compared',
            automatic_promotion=False,formal_dispatched=False))
    except BaseException as exc:
        write_new(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True)); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    import torch
    torch.set_num_threads(1);run(c,provenance)
