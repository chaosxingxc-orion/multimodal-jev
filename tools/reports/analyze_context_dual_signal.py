"""Split archived GRPO constant groups into all-correct and all-wrong.

Fixed TRAIN diagnostic only. No resampling, model calls, optimizer, or target
data. Binds every consumed group to the archived independent audit/transport;
does not rerun the full model/runtime audit or certify effective gradients.
"""
import argparse
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.run_context_batch_probe_candidate import verify


def branch_summary(branch):
    rewards=branch['rewards'];actions=branch['actions'];target=branch['target']
    p=branch['probability'];weight=branch['weight'];adv=branch['advantages']
    if (rewards.ndim!=2 or rewards.shape[1]!=16 or not rewards.shape[0]
            or actions.shape!=rewards.shape or adv.shape!=rewards.shape
            or target.shape!=(len(rewards),) or target.dtype!=torch.bool
            or p.shape!=target.shape or weight.shape!=target.shape
            or not ((actions==0)|(actions==1)).all()
            or not torch.equal(rewards,(actions==target[:,None]).float())
            or not torch.isfinite(p).all() or not ((p>0)&(p<1)).all()
            or not torch.isfinite(weight).all() or (weight<0).any()
            or not torch.isclose(weight.sum(),torch.tensor(1.,dtype=weight.dtype))):
        raise ValueError('complete G16 binary component rewards and normalized weights required')
    centered=rewards-rewards.mean(1,keepdim=True)
    expected=centered/centered.square().mean(1,keepdim=True).sqrt().clamp_min(.25)
    if not torch.allclose(adv,expected,atol=2e-7,rtol=0):
        raise ValueError('recorded advantages differ from floor-clipped GRPO')
    masks=dict(all_correct=(rewards==1).all(1),all_wrong=(rewards==0).all(1))
    masks['mixed']=~(masks['all_correct']|masks['all_wrong'])
    success=torch.where(target,p.double(),1-p.double())
    expected_mixed=1-success.pow(16)-(1-success).pow(16)
    def summarize(mask):
        return dict(components=int(mask.sum()),weight_mass=float(weight[mask].sum()),
            expected_mixed_components=float(expected_mixed[mask].sum()),
            counts={k:int((v&mask).sum()) for k,v in masks.items()},
            weight={k:float(weight[v&mask].sum()) for k,v in masks.items()})
    return dict(all=summarize(torch.ones_like(target)),positive=summarize(target),negative=summarize(~target))


def aggregate(rows):
    if not rows:raise ValueError('nonempty real query rows required')
    result={}
    for name in ('ranking','existence'):
        total={}
        for label in ('all','positive','negative'):
            values=[r['branches'][name][label] for r in rows]
            count=sum(v['components'] for v in values)
            mass=sum(v['weight_mass'] for v in values)
            counts={k:sum(v['counts'][k] for v in values) for k in ('all_correct','all_wrong','mixed')}
            weights={k:sum(v['weight'][k] for v in values) for k in counts}
            total[label]=dict(components=count,counts=counts,
                fractions={k:v/count if count else None for k,v in counts.items()},
                mean_weight_mass_per_query={k:v/len(rows) for k,v in weights.items()},
                weight_fractions={k:v/mass if mass else None for k,v in weights.items()},
                expected_mixed_fraction=sum(v['expected_mixed_components'] for v in values)/count if count else None)
        result[name]=total
    def any_wrong(r):return any(r['branches'][b]['all']['counts']['all_wrong'] for b in result)
    def any_mixed(r):return any(r['branches'][b]['all']['counts']['mixed'] for b in result)
    return dict(query_exposures=len(rows),branches=result,queries=dict(
        all_components_sampled_correct=sum(not any_wrong(r) and not any_mixed(r) for r in rows),
        has_all_wrong_component=sum(bool(any_wrong(r)) for r in rows),
        has_mixed_component=sum(bool(any_mixed(r)) for r in rows),
        has_wrong_but_no_mixed=sum(bool(any_wrong(r)) and not any_mixed(r) for r in rows)))


def analyze(c):
    base=Path(c['operations_root']);package=Path(c['prepared'])/'package'
    if verify(package)!=c['manifest_sha256']:raise ValueError('sealed training package differs')
    training=json.loads((package/'training.json').read_text())
    if (training['experiment_kind']!='efficiency' or training['start_update']!=0
            or training['stop_update']!=100 or training['stage']!='task'
            or training['algorithm']!='dual_credit_grpo_floor_clipped_v1' or training['dual_arm']!='joint'):
        raise ValueError('complete100 TRAIN joint GRPO diagnostic required')
    audit_path=base/'audit.json';transport_path=base/'transport.json'
    if file_hash(audit_path)!=c['audit_sha256']:raise ValueError('independent audit binding changed')
    audit=json.loads(audit_path.read_text());transport=json.loads(transport_path.read_text())
    if (audit['status']!='finite_dp2_efficiency_audited' or len(audit['updates'])!=100
            or audit['files_sha256']!=c['manifest_sha256']
            or audit['transport_sha256']!=file_hash(transport_path)):
        raise ValueError('matching complete archived independent audit required')
    rows=[];hashes={};steps=[]
    for index,proof in enumerate(audit['updates']):
        name=f'run/groups-{index:04d}.pt';path=base/'collected'/name
        digest=file_hash(path)
        if path.is_symlink() or transport['files'].get(name)!=digest:raise ValueError('archived group bytes changed')
        hashes[name]=digest;groups=torch.load(path,map_location='cpu',weights_only=True);current=[]
        if proof['update']!=index or len(groups)!=proof['checked_slots']:
            raise ValueError('audited update cursor or group coverage differs')
        for group in groups:
            if not group['real_request']:continue
            binding=group['dual_credit']['binding']
            if (binding['update']!=index or binding['algorithm']!=training['algorithm']
                    or binding['group_size']!=16 or binding['query_id']!=group['request_id']):
                raise ValueError('query credit binding differs')
            current.append(dict(update=index+1,request_id=group['request_id'],
                branches={b:branch_summary(group['dual_credit']['branches'][b]) for b in ('ranking','existence')}))
        if len(current)!=proof['real_queries']:raise ValueError('real query denominator differs')
        step=aggregate(current)
        for b in ('ranking','existence'):
            actual=step['branches'][b]['all'];expected=proof['branches'][b]
            if (actual['components']!=expected['components'] or
                    actual['counts']['all_correct']+actual['counts']['all_wrong']!=expected['zero_signal_components']):
                raise ValueError('signal totals differ from independent recorded audit')
        steps.append(dict(update=index+1,**step));rows.extend(current)
    return dict(schema='jev-dual-signal-diagnostic-v1',all_updates=aggregate(rows),
        windows=[dict(start_update=i+1,stop_update=i+10,**aggregate([r for r in rows if i<r['update']<=i+10]))
                 for i in range(0,100,10)],steps=steps,
        source=dict(manifest_sha256=c['manifest_sha256'],audit_sha256=c['audit_sha256'],
            transport_sha256=audit['transport_sha256'],group_files=hashes,script_sha256=file_hash(__file__)),
        runtime_reaudited=False,resampling_performed=False,
        scope='Archived TRAIN exposures under evolving policies, not fixed-policy dataset difficulty or heldout performance. '
              'Positive/negative ranking targets depend on canonical pair ordering. '
              'Constant-wrong sampled components may have auxiliary/KL gradients. Query strata overlap; '
              'expected G16 mixing is conditional on saved probabilities, not a larger-group speed claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);torch.set_num_threads(1)
    result=analyze(c);result['config_provenance']=provenance
    output=Path(c['output']);check_paths({'signal_report':output});write_new(output,result)
    print(json.dumps(result['all_updates'],indent=2))
