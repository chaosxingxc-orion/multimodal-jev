import hashlib
import json

import pytest

from tools.reports import compare_simple_endpoints as scores


def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(tmp_path,policy,prob,*,flip_label=False):
    name='toy'; root=tmp_path/policy
    plan=dict(schema='jev-stage2-view-selection-source-v1',evaluation_dataset=name,target_tuning_allowed=False,
              model_selection_performed=False,checkpoints=[dict(step=0),dict(step=2,policy_id=policy)],
              base_id='b',release=dict(sha256='1'),training_audit=dict(sha256=policy))
    p=write(root/'source-plan.json',plan)
    files={'source-plan.json':p['sha256']}
    targets=[dict(id=str(i),group=str(i),audio_sha256=str(i),gold='a' if i%2==0 else 'b') for i in range(8)]
    if flip_label:targets.reverse()
    ids=['a','b']
    rows=[dict(id=t['id'],audio_sha256=t['audio_sha256']) for t in targets]
    for rel,v in [(f'inference/{name}/panel.json',dict(candidate_ids=ids)),
                  (f'inference/{name}/rows.json',rows),(f'scoring/{name}/targets.json',targets)]:
        files[rel]=write(root/rel,v)['sha256']
    meta=dict(split='source_selection',recordings=8,candidates=2)
    seal=write(root/'MANIFEST.json',dict(schema='jev-native-evaluation-seal-v1',datasets={name:meta},
               target_tuning_allowed=False,files=files))
    predictions=[dict(id=t['id'],request_identity=dict(policy_id=policy),policy_id=policy,
                      prob=[prob if label==t['gold'] else 1-prob for label in ids]) for t in targets]
    pred=write(root/'pred.json',dict(records=predictions))
    import math
    metrics=dict(candidate_ids=ids,queries=8,top1_macro_f1=float(prob>.5),top1_event_accuracy=float(prob>.5),
                 candidate_binary_nll=-math.log(prob),candidate_brier=(1-prob)**2,mean_top1_probability=max(prob,1-prob))
    audit=write(root/'audit.json',dict(policy_id=policy,status='complete_macro_evaluation_outputs_audited',
                recordings=8,decisions=2,step=2,seal_sha256=seal['sha256'],metrics=metrics,
                predictions_sha256=pred['sha256'],dataset=name,target_tuning_allowed=False))
    return dict(plan=p,seal=dict(root=str(root),sha256=seal['sha256']),predictions=pred['path'],audit=audit)


def test_paired_improvement_and_exact_baseline_lock(tmp_path):
    spec=dict(recordings=8,classes=2,split='source_selection',cluster='source_group')
    ref=lambda key,p:dict(policy_id=key,cells={'toy':fixture(tmp_path,key,p)})
    design=write(tmp_path/'design.json',dict(interval=dict(method='paired_cluster_percentile_bootstrap',
                     confidence=.95,samples=100,seed=1)))
    c=dict(schema='jev-simple-endpoint-comparison-v1',mode='bind_baseline',baseline=ref('left',.2),
           candidate=None,execution=dict(cpu_workers=1,window=1,min_parallel=1,chunk=1),
           design=design,datasets=dict(toy=spec))
    baseline=scores.run(c)
    c.update(mode='compare',baseline_receipt=write(tmp_path/'baseline.json',baseline),candidate=ref('right',.9))
    paired=scores.run(c)['datasets']['toy']
    assert paired['improves_with_probability_quality'] and paired['paired']['fixed']==8
    c['baseline']=ref('wrong',.2)
    with pytest.raises(ValueError,match='preregistered'):scores.run(c)


def test_reordered_roster_and_tampered_predictions_rejected(tmp_path):
    l=fixture(tmp_path,'left',.3); r=fixture(tmp_path,'right',.7,flip_label=True)
    spec=dict(recordings=8,classes=2,split='source_selection',cluster='source_group')
    with pytest.raises(ValueError,match='identical native panel'):
        scores.evaluate(('toy',spec,(l,'left'),(r,'right'),{}))
    with open(l['predictions'],'a') as f:f.write(' ')
    with pytest.raises(ValueError,match='changed'):scores.load_cell('toy',spec,l,'left')
