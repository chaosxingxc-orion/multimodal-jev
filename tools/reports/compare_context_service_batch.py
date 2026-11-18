"""Compare complete100 DP2 runs changing only service32/token capacity."""
import json
from pathlib import Path

from tools.reports.context_macro_efficiency_summary import summarize
from tools.reports.prepare_context_dual_replay import relocate


def controls(ops):
    root=Path(ops['prepared'])/'package'
    t=relocate(json.loads((root/'training.json').read_text()),ops['remote_root'],'/comparison')
    services=[];capacities=[]
    for i in range(2):
        s=relocate(json.loads((root/f'service-{i}.json').read_text()),ops['remote_root'],'/comparison')
        capacities.append((s['policy'].pop('max_batch_queries'),s['policy'].pop('max_batch_tokens')))
        services.append(s)
    if (t['microbatch']!=8 or t['global_batch']!=128 or t['start_update']!=0
            or t['stop_update']!=100 or t['experiment_kind']!='efficiency'):
        raise ValueError('fixed micro8/global128 complete100 comparison required')
    return (t,services),capacities


def compare(baseline,candidate,audits):
    a,ac=controls(baseline);b,bc=controls(candidate)
    if a!=b or ac!=[(4,16384)]*2 or bc!=[(32,131072)]*2:
        raise ValueError('training or service differences exceed declared capacity change')
    for ops,audit in zip((baseline,candidate),audits,strict=True):
        if (audit['status']!='finite_dp2_efficiency_audited'
                or audit['files_sha256']!=ops['manifest_sha256']):
            raise ValueError('complete independently audited run required')
    if audits[0]['image']!=audits[1]['image']:raise ValueError('same image required')
    stats=[];records=[]
    for ops in (baseline,candidate):
        run=Path(ops['operations_root'])/'collected/run'
        stats.append(summarize(run,start_update=0,stop_update=100,warmup_updates=2))
        records.append([json.loads((run/f'batch-{i:04d}.json').read_text())['records'] for i in range(100)])
    if records[0]!=records[1] or [s['reference_misses'] for s in stats[0]['steps']]!=[s['reference_misses'] for s in stats[1]['steps']]:
        raise ValueError('actual input order or reference cache condition differs')
    return dict(schema='jev-service-batch-efficiency-comparison-v1',baseline=stats[0],candidate=stats[1],
        all_updates_speedup=stats[0]['all_updates']['complete_seconds']/stats[1]['all_updates']['complete_seconds'],
        after_warmup_speedup=stats[0]['measured']['complete_seconds']/stats[1]['measured']['complete_seconds'],
        saved_complete_seconds=stats[0]['all_updates']['complete_seconds']-stats[1]['all_updates']['complete_seconds'],
        source_audits=audits,matched_actual_queries=True,matched_reference_misses=True,
        changed=['service batch4->32','token capacity16384->131072'],
        scope='Complete optimizer updates including activation; excludes startup and collection. No efficacy or transfer claim.')
