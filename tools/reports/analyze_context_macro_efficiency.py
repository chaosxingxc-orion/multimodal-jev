"""Fixed analysis of audited complete update costs and GRPO signal density.

No training mutation or target access. Serial savings are counterfactual bounds,
not measured speedups. Component signal density is not query success or gradient.
"""
import argparse
import json
import math
from pathlib import Path

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.audit_context_macro_training import audit_collected
from tools.reports.context_macro_efficiency_summary import summarize


def analyze(summary,audit):
    rows=summary['steps'];updates=audit['updates']
    if (audit['status'] not in ('finite_dp2_training_audited','finite_dp2_efficiency_audited','finite_dp2_replay_audited')
            or len(rows)!=len(updates) or not rows):
        raise ValueError('complete independently audited update window required')
    components=[];branches={b:dict(components=0,zero_signal_components=0,weighted_reward_numerator=0.)
                            for b in ('ranking','existence')}
    for row,proof in zip(rows,updates,strict=True):
        if row['update']!=proof['update']+1 or row['real_queries']!=proof['real_queries']:
            raise ValueError('audited step and real query identity differ')
        preparation=row['rank0_preparation_seconds'];rollout=row['rollout_seconds']
        reference=row['reference_seconds'];behavior=row['behavior_and_credit_seconds']
        actor=row['rank0_actor_seconds'];export=row['rank0_export_checkpoint_sync_seconds']
        phases=dict(preparation_outside_rollout=preparation-rollout,reference=reference,
            behavior_and_credit=behavior,rollout_other=rollout-reference-behavior,
            actor=actor,export_checkpoint_sync=export,
            rank_completion_gap=row['complete_seconds']-preparation-actor-export)
        if any(not math.isfinite(v) or v < -1e-6 for v in phases.values()):
            raise ValueError('nested timers cannot form a nonnegative partition')
        phases={k:max(0.,v) for k,v in phases.items()}
        if not math.isclose(sum(phases.values()),row['complete_seconds'],abs_tol=1e-5):
            raise ValueError('phase costs must partition complete step time')
        components.append(phases)
        for name in branches:
            signal=proof['branches'][name];count=signal['components'];zero=signal['zero_signal_components']
            reward=signal['mean_weighted_reward']
            if (type(count) is not int or type(zero) is not int or not 0<=zero<=count or count<=0
                    or type(reward) not in (int,float) or not math.isfinite(reward)):
                raise ValueError('finite reward and valid component signal counts required')
            branches[name]['components']+=count;branches[name]['zero_signal_components']+=zero
            branches[name]['weighted_reward_numerator']+=reward*count
    def window(start):
        selected=rows[start:];parts=components[start:]
        total=sum(r['complete_seconds'] for r in selected)
        exposures=sum(r['real_queries'] for r in selected)
        totals={k:sum(p[k] for p in parts) for k in parts[0]}
        costs={k:dict(seconds=v,fraction=v/total,
            counterfactual_speedup_if_halved=total/(total-v*.5),
            counterfactual_speedup_if_eliminated=None if total==v else total/(total-v)) for k,v in totals.items()}
        return dict(updates=len(selected),real_query_exposures=exposures,complete_seconds=total,
            queries_per_second=exposures/total,
            four_device_update_window_hours=4*total/3600,
            peak_actor_allocated_bytes=max(r['peak_actor_allocated_bytes'] for r in selected),
            phases=costs,largest_phases=sorted(costs,key=lambda k:-costs[k]['seconds']))
    for stats in branches.values():
        count=stats['components'];nonzero=count-stats['zero_signal_components']
        stats['nonzero_signal_components']=nonzero
        stats['nonzero_signal_fraction']=nonzero/count
        stats['mean_weighted_reward']=stats.pop('weighted_reward_numerator')/count
        stats['nonzero_components_per_update_second']=nonzero/sum(r['complete_seconds'] for r in rows)
    return dict(schema='context-macro-efficiency-analysis-v1',all_updates=window(0),
        after_warmup=window(summary['warmup_updates']),all_update_signal=branches,
        counterfactuals_are_measured_speedups=False,
        scope='Audited finite update window; rank0 nested timers partitioned plus rank completion gap. '
              'Four-device hours cover update windows only, excluding startup/collection/evaluation. '
              'Signal counts are components, not queries, task accuracy or effective gradients. '
              'Microbatch changes and reference caching require separate matched experiments.')


def run(config,provenance):
    operations,_=load_config(config['operations_config'],config['operations_local'])
    audit=audit_collected(operations)
    training=json.loads((Path(operations['prepared'])/'package/training.json').read_text())
    summary=summarize(Path(operations['operations_root'])/'collected/run',
        start_update=training['start_update'],stop_update=training['stop_update'],warmup_updates=config['warmup_updates'])
    result=analyze(summary,audit)
    result['source']=dict(manifest_sha256=audit['files_sha256'],transport_sha256=audit['transport_sha256'],
        containers=audit['containers'],image=audit['image'],timer_files=summary['files'])
    result['config_provenance']=provenance
    output=Path(config['output']);output.parent.mkdir(parents=True,exist_ok=True);commit_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    result=run(config,provenance)
    print(json.dumps(dict(all_updates=result['all_updates'],signal=result['all_update_signal']),indent=2))
