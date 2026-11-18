"""Native attribute metrics; input/runtime authentication is a caller obligation.

Never repairs Final using attributes. Process scores cannot choose checkpoints.
Synthetic controls exercise this scorer without claiming model results.
"""
import math
import torch
from multimodal_jev.context_event_metrics import ContextEventMetrics
from tools.reports.context_attribute_reward_candidate import layout, validate_truth
from tools.reports.build_context_attribute_candidate import digest

STEPS=[0,*range(64,634,64),634]
SLICES=('at_least_one_attribute_unique','both_attributes_ambiguous')


def score_records(inputs,targets,predictions,menu,*,arm,policy_id,expected_count):
    """Score aligned complete supplied population, never certify its provenance."""
    nodes,indices=layout(menu,arm);ids=[n['id'] for n in nodes]
    all_ids=[n['id'] for n in menu['nodes']]
    if (type(expected_count) is not int or expected_count<1 or len(inputs)!=expected_count
            or len(targets)!=expected_count or len(predictions)!=expected_count
            or len({r['id'] for r in inputs})!=expected_count or not isinstance(policy_id,str) or not policy_id):
        raise ValueError('Complete unique aligned population and explicit policy required')
    groups={};details=[];source_truth={}
    for inp,target,pred in zip(inputs,targets,predictions,strict=True):
        if (target['id']!=inp['id'] or pred['id']!=inp['id'] or target['input_sha256']!=digest(inp)
                or inp['catalog_sha256']!=digest(menu) or target['candidate_ids']!=all_ids
                or pred['decision_ids']!=ids or pred['policy_id']!=policy_id or pred['arm']!=arm
                or target['structural_slice'] not in SLICES
                or not isinstance(target['source_group'],str) or not target['source_group']
                or len(target['labels'])!=len(all_ids) or len(target['supervised'])!=len(all_ids)
                or any(type(v) is not bool for v in target['labels']+target['supervised'])):
            raise ValueError('Aligned input, truth, candidate order and policy required')
        labels=torch.tensor([target['labels']],dtype=torch.bool)
        observed=torch.tensor([target['supervised']],dtype=torch.bool)
        validate_truth(labels,observed,menu,'q_input')
        prob=pred['prob']
        if len(prob)!=len(ids) or any(type(p) not in (float,int) or not math.isfinite(p) or not 0<=p<=1 for p in prob):
            raise ValueError('Complete finite probabilities required')
        truth={k:v for k,v in zip(all_ids,target['labels'],strict=True)}
        expected_slice='both_attributes_ambiguous' if all(len(n['members'])>1 for n in menu['nodes']
            if n['family']!='final' and truth[n['id']]) else 'at_least_one_attribute_unique'
        if target['structural_slice']!=expected_slice:raise ValueError('Native structural slice differs')
        gold=next(n['id'] for n in menu['nodes'] if n['family']=='final' and truth[n['id']])
        if source_truth.setdefault(target['source_group'],gold)!=gold:raise ValueError('Conflicting source group truth')
        for family,index in indices.items():
            if not index:continue
            candidate_ids=[ids[i] for i in index];values=[prob[i] for i in index]
            assigned=next(k for k in candidate_ids if truth[k])
            for view in ('all',target['structural_slice']):
                key=(view,family)
                if key not in groups:groups[key]=dict(metric=ContextEventMetrics(candidate_ids),empty=0,multiple=0,n=0)
                group=groups[key];group['metric'].add(inp['id'],assigned,values)
                count=sum(v>=.5 for v in values)
                group['empty']+=count==0;group['multiple']+=count>1;group['n']+=1
        fidx=indices['final'];chosen=[ids[i] for i in fidx if prob[i]>=.5]
        top=ids[max(fidx,key=lambda i:prob[i])]
        details.append(dict(id=inp['id'],source_group=target['source_group'],gold=gold,
            structural_slice=expected_slice,set_f1=2*int(gold in chosen)/(len(chosen)+1),
            exact_set=chosen==[gold],top1_correct=top==gold,selected_count=len(chosen)))
    reports={}
    for (view,family),g in groups.items():
        reports.setdefault(view,{})[family]=dict(g['metric'].result(),
            empty_selection_rate=g['empty']/g['n'],multiple_selection_rate=g['multiple']/g['n'])
    return dict(arm=arm,policy_id=policy_id,recordings=expected_count,source_groups=len(source_truth),
        metrics=reports['all'],structural_slices={s:reports.get(s) for s in SLICES},records=details,
        scope='Supplied aligned population scoring only; caller must authenticate release, processor and actual model predictions')


def select_curve(points):
    """Complete predeclared curve; select solely using Final F1/NLL/step."""
    if len(points)!=len(STEPS) or sorted(p['step'] for p in points)!=STEPS:
        raise ValueError('Complete unique 11-point curve required')
    first=points[0]['score']
    roster=[{k:r[k] for k in ('id','source_group','gold','structural_slice')} for r in first['records']]
    for p in points:
        score=p['score'];f=score['metrics']['final']
        if score['arm']!=first['arm'] or score['recordings']!=first['recordings'] or [
                {k:r[k] for k in ('id','source_group','gold','structural_slice')} for r in score['records']]!=roster:
            raise ValueError('Same complete population and arm for every checkpoint required')
        if (type(f['query_macro_set_f1']) not in (int,float) or not math.isfinite(f['query_macro_set_f1'])
                or not 0<=f['query_macro_set_f1']<=1 or
                (f['candidate_binary_nll'] is None)!=bool(f['binary_nll_infinite'])
                or f['candidate_binary_nll'] is not None and (not math.isfinite(f['candidate_binary_nll']) or f['candidate_binary_nll']<0)):
            raise ValueError('Valid Final score and explicit infinite NLL required')
    def key(p):
        f=p['score']['metrics']['final'];nll=f['candidate_binary_nll']
        return (-f['query_macro_set_f1'],float('inf') if nll is None else nll,p['step'])
    chosen=min(points,key=key)
    return dict(step=chosen['step'],policy_id=chosen['score']['policy_id'],arm=first['arm'],
        rule='max_Final_query_macro_set_F1_then_min_Final_binary_NLL_then_earliest_step',
        process_metrics_used=False)


def paired_delta(a,b):
    """Recording-weighted paired point differences, no significance claim."""
    if len(a['records'])!=len(b['records']) or not a['records']:raise ValueError('Complete pair required')
    for x,y in zip(a['records'],b['records'],strict=True):
        if any(x[k]!=y[k] for k in ('id','source_group','gold','structural_slice')):
            raise ValueError('Identical source-group/recording pairing required')
    return dict(recordings=len(a['records']),
        final_set_f1_delta=sum(y['set_f1']-x['set_f1'] for x,y in zip(a['records'],b['records'],strict=True))/len(a['records']),
        top1_improved=sum(not x['top1_correct'] and y['top1_correct'] for x,y in zip(a['records'],b['records'],strict=True)),
        top1_regressed=sum(x['top1_correct'] and not y['top1_correct'] for x,y in zip(a['records'],b['records'],strict=True)),
        scope='Paired descriptive comparison; no bootstrap interval, seed variance or causal reasoning claim')
