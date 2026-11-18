"""Read archived timers for replica choice; no live polling or topology mutation.

Single-engine historical probe and macro probe have different measurement
boundaries. Report their ratio as descriptive, never a matched training speedup.
"""
import argparse
import json
from pathlib import Path
import statistics

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json


def analyze(c):
    hashes={}
    def reader(root,transport):
        root=Path(root);transport=Path(transport)
        files=json.loads(transport.read_text())['files']
        hashes[str(transport)]=file_hash(transport)
        def read(name):
            path=root/name
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('contained archived evidence required')
            digest=file_hash(path)
            if files.get(name)!=digest:raise ValueError('archived evidence changed: '+name)
            hashes[str(path)]=digest
            return json.loads(path.read_text())
        return read
    single=reader(c['single_collection'],c['single_transport'])
    dual=reader(c['dual_collection'],c['dual_transport'])
    training=reader(c['training_collection'],c['training_transport'])
    one=single('evaluation/batched/measurement.json')
    two=dual('measurement/measurement.json')
    old=single('config.json')
    def timed(report):
        values=[p['seconds'] for p in report['passes'] if p['kind']=='timed']
        if len(values)!=2 or any(v<=0 for v in values):raise ValueError('two historical timed passes required')
        return statistics.mean(values)
    if one['rows']!=128 or two['rows']!=128 or one['policy_id']!=two['policy_id']:
        raise ValueError('same initial policy and 128 inputs required')
    for a,b in zip(one['passes'],two['passes'],strict=True):
        if a['records']!=b['records'] or a['sampling']!=b['sampling']:
            raise ValueError('historical actual input/logits/probability/sample identities differ')
    one_seconds=timed(one);two_seconds=timed(two)
    result=training('run/result.json')
    if result['start_update']!=0 or result['stop_update']!=2 or not result['passed']:
        raise ValueError('completed two-update diagnostic required')
    macros=[]
    for index in range(4):
        prefix=c['journal_prefix']+f'/batch-{index:08d}'
        manifest=training(prefix+'/INPUT.json');done=training(prefix+'/COMPLETE.json')
        loads=[0.,0.];counts=[0,0]
        for job in manifest['jobs']:
            row=training(prefix+'/'+job['rollout_id']+'.json')
            if any(row[k]!=v for k,v in job.items()) or row['seconds']<=0:
                raise ValueError('complete matching rollout job required')
            replica=row['replica_id'];loads[replica]+=row['seconds'];counts[replica]+=len(row['indices'])
        wall=done['seconds'];gap=wall-max(loads)
        if sum(counts)!=128 or gap<0:raise ValueError('complete nonnegative macro timing required')
        macros.append(dict(update=index//2+1,phase='reference' if index%2==0 else 'behavior',
            macro_seconds=wall,replica_seconds=loads,replica_queries=counts,
            outer_gap_seconds=gap,outer_gap_fraction=gap/wall,
            task_time_balance=sum(loads)/(2*max(loads)),
            serial_task_time_counterfactual=sum(loads),
            scope='Per-job clocks include IPC, model execution and validation; gap is not all RPC overhead.'))
    steps=[]
    for index in range(2):
        ranks=[training(f'run/rank-{rank}/update-{index+1:04d}.json') for rank in range(2)]
        x=ranks[0];r=x['rollout'];ref,behavior=macros[index*2:index*2+2]
        steps.append(dict(update=index+1,total_seconds=max(v['total_seconds'] for v in ranks),
            actor_seconds=x['actor_seconds'],preparation_seconds=x['preparation_seconds'],
            reference_phase_seconds=r['reference_seconds'],behavior_credit_seconds=r['behavior_and_credit_seconds'],
            reference_outside_macro_seconds=r['reference_seconds']-ref['macro_seconds'],
            behavior_credit_outside_macro_seconds=r['behavior_and_credit_seconds']-behavior['macro_seconds'],
            rollout_other_seconds=r['total_seconds']-r['reference_seconds']-r['behavior_and_credit_seconds'],
            export_checkpoint_sync_seconds=x['total_seconds']-x['preparation_seconds']-x['actor_seconds'],
            switches=len(r['switches']),reference_misses=r['reference_misses']))
    return dict(schema='context-replica-overhead-analysis-v1',historical_probe=dict(
        single_engine_batch4_seconds=one_seconds,dual_engine_batch4_macro_seconds=two_seconds,
        descriptive_ratio=one_seconds/two_seconds,input_sha256=old['input_sha256'],
        exact_record_and_G16_equivalence=True,matched_end_to_end_training_comparison=False),
        macros=macros,steps=steps,evidence_sha256=hashes,
        recommendation='retain_two_TP1_instances_for_wall_time_pending_matched_1_vs_2_update_trial',
        device_cost_break_even='With actor DP2 fixed, two replicas use fewer allocated device-seconds only if T_single/T_dual > 4/3.',
        unmeasured=['RPC serialization vs kernel cost','isolated activation wall time','inference allocated memory in current training',
                    'same-service matched single-replica full updates','warm-reference-cache topology comparison'],
        scope='Hash-checked archived evidence, not a fresh full runtime audit or measured training topology speedup.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    r=analyze(c);r['provenance']=provenance;commit_json(c['output'],r)
    print(json.dumps({k:v for k,v in r.items() if k!='evidence_sha256'},indent=2))
