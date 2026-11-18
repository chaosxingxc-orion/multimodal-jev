"""Reconstruct complete acoustic/KV matched100 evidence; never deploy a model."""
import argparse
import json
import math
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports import context_macro_training_operations as baseline
from tools.reports import context_acoustic_training_operations as acoustic
from tools.reports import context_kv_training_operations as kv
from tools.reports.audit_context_macro_training import audit_collected
from tools.reports.run_context_acoustic_training_study import compare as acoustic_compare
from tools.reports.run_context_kv_training_study import compare as kv_compare

ARMS = ('service32', 'acoustic', 'acoustic_kv16')
STUDIES = (('acoustic', 'acoustic_training100_compared'), ('kv', 'kv_training100_compared'))


def read(path): return json.loads(Path(path).read_text())


def choose(comparisons):
    a, k = comparisons
    for comparison, schema in zip(comparisons, ('jev-acoustic-training-efficiency-v1',
                                               'jev-kv-training-efficiency-v1'), strict=True):
        if (comparison.get('schema') != schema or comparison.get('matched_actual_queries') is not True
                or comparison.get('matched_reference_misses') is not True
                or len(comparison.get('source_audits', [])) != 2
                or any(v.get('status') != 'finite_dp2_efficiency_audited' for v in comparison['source_audits'])):
            raise ValueError('two complete independently audited matched100 comparisons required')
    if (a['candidate'] != k['baseline']
            or a['source_audits'][1]['files_sha256'] != k['source_audits'][0]['files_sha256']):
        raise ValueError('same acoustic bridge run required')
    stats = dict(zip(ARMS, (a['baseline'], a['candidate'], k['candidate']), strict=True))
    for s in stats.values():
        if s['optimizer_steps'] != 100 or s['warmup_updates'] != 2:
            raise ValueError('fixed100 with two warmup updates required')
        for key, count in (('all_updates', 100), ('measured', 98)):
            w = s[key]; seconds = w['complete_seconds']
            if (w['updates'] != count or w['real_query_exposures'] != 128*count
                    or type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0):
                raise ValueError('complete positive finite matched-window timers required')
    # Descriptive ranking only: ties retain the earlier, simpler configuration.
    fastest = min(ARMS, key=lambda n:stats[n]['all_updates']['complete_seconds'])
    warm = min(ARMS, key=lambda n:stats[n]['measured']['complete_seconds'])
    return dict(fastest_observed_arm=fastest, fastest_after_warmup_arm=warm,
        window_rankings_agree=fastest==warm,
        comparisons=[dict(baseline=left,candidate=right,
            throughput_ratio=stats[left]['all_updates']['complete_seconds']/stats[right]['all_updates']['complete_seconds'],
            saved_seconds=stats[left]['all_updates']['complete_seconds']-stats[right]['all_updates']['complete_seconds'])
            for left,right in (('service32','acoustic'),('acoustic','acoustic_kv16'),('service32','acoustic_kv16'))],
        arms={n:dict(all_updates=s['all_updates'],measured=s['measured']) for n,s in stats.items()},
        rule='descriptive_minimum_complete100_seconds_then_simpler_runtime',
        automatic_promotion=False, formal_training_dispatch_allowed=False,
        limitations=['single sequential run per arm; causal attribution and repeatability not established',
                     'startup/evaluation and inference memory excluded from update timers',
                     'matching selected runtime tail/parity admission still required',
                     'task efficacy and transfer must be evaluated independently'])


def report(c):
    paths = [*c['operations'].values(), *c['studies'].values()]
    if set(c['operations']) != set(ARMS) or set(c['studies']) != {'acoustic','kv'}:
        raise ValueError('explicit three arms and two studies required')
    if not set(paths) <= set(c['pins']): raise ValueError('all configuration inputs must be pinned')
    for path, digest in c['pins'].items():
        if file_hash(path) != digest: raise ValueError('recommendation dependency changed: '+path)
    ops = {name:read(path) for name,path in c['operations'].items()}
    if ops['acoustic']['parent_operations'] != ops['service32'] or ops['acoustic_kv16']['parent_operations'] != ops['acoustic']:
        raise ValueError('exact configuration ancestry required')
    pending=[]; failed=[]; evidence={}
    for name, status in STUDIES:
        study=read(c['studies'][name]); arm='acoustic' if name=='acoustic' else 'acoustic_kv16'
        if read(study['operations']) != ops[arm]: raise ValueError('study must execute declared arm')
        root=Path(study['output']); terminal=root/'terminal.json'
        if not terminal.exists():
            pending.append(name); continue
        state=read(terminal); evidence[str(terminal)]=file_hash(terminal)
        if state.get('exit_code') != 0 or state.get('status') != status:
            failed.append(dict(study=name,terminal=state)); continue
        if not (root/'comparison.json').is_file(): raise ValueError('successful study missing comparison')
    if pending or failed:
        return dict(schema='jev-training-efficiency-recommendation-v1',
            status='failed_evidence_preserved' if failed else 'awaiting_complete_evidence',
            pending_studies=pending,failed_studies=failed,evidence=evidence,
            formal_training_dispatch_allowed=False,automatic_promotion=False)
    audits=[]
    for name, module in zip(ARMS,(baseline,acoustic,kv),strict=True):
        audits.append(audit_collected(ops[name],validator=module.validate,
            command_builder=module.command,output_auditor=acoustic.audit_outputs))
    comparisons=[acoustic_compare(ops['service32'],ops['acoustic'],audits[:2]),
                 kv_compare(ops['acoustic'],ops['acoustic_kv16'],audits[1:])]
    for (name,_), result in zip(STUDIES,comparisons,strict=True):
        path=Path(read(c['studies'][name])['output'])/'comparison.json'
        if result != read(path): raise ValueError('independent reconstruction differs: '+name)
        evidence[str(path)]=file_hash(path)
    return dict(schema='jev-training-efficiency-recommendation-v1',status='complete_efficiency_evidence_reconstructed',
        evidence=evidence,manifests={n:o['manifest_sha256'] for n,o in ops.items()},**choose(comparisons))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    args=p.parse_args();c,provenance=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    result=report(c);result['config_provenance']=provenance
    output=Path(c['output']);check_paths({'recommendation':output});output.parent.mkdir(parents=True,exist_ok=True)
    write_new(output,result);print(json.dumps(result,indent=2))
