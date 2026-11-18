import copy
import json

import pytest

from tools.jev import compare_context_rl_curves as module
from tools.jev.context_selection_curve import summarize


def arms():
    steps=[0,*range(64,634,64),634]
    fields=['query_macro_set_f1','top1_event_accuracy','candidate_binary_nll','candidate_brier']
    experiment=dict(release_sha256='data',selection_recordings=9226,task_stage=dict(updates=634),
        evaluation=dict(checkpoints=steps,primary=fields[0],secondary=fields[1:],set_threshold=.5))
    result=[]
    for algorithm in ('grpo_v1','rloo_v1'):
        points=[]
        for i,step in enumerate(steps):
            value=.5 + (.01 if algorithm=='grpo_v1' else .005)*i
            if step==634:value-=.02  # Preserve a final regression instead of selecting it away.
            metrics=dict(zip(fields,[value,.6,.2,.1]))
            metrics.update(queries=9226,threshold=.5,candidate_ids=[str(k) for k in range(60)],
                binary_nll_infinite=False,set_precision_defined_queries=9226)
            points.append(dict(step=step,result=dict(metrics=metrics,policy_id=algorithm+str(step),
                role='selection',release_sha256='data',predictions_sha256='prediction')))
        curve=summarize(points,experiment,initial_policy_id=algorithm+'0')
        curve.update(algorithm=algorithm,run_id=algorithm,initial_source_policy_id='stage1',
            selected_source_checkpoint=dict(step=curve['selected_checkpoint']['step'],policy_id='selected'))
        updates=[dict(update=i,real_queries=65 if i in (316,633) else 128,
            zero_signal_groups=1,slowest_rank_step_seconds=10.,mean_reward=.3,
            mean_group_reward_std=.1,mean_abs_advantage=.8) for i in range(634)]
        result.append(dict(curve=curve,training=dict(container_id=algorithm,updates=updates,
            state=dict(StartedAt='2026-11-15T00:00:00.123456789Z',FinishedAt='2026-11-15T04:00:00.123456789Z')),
            experiment=experiment,config={'microbatch':8},source_files={'source/trainer.py':'same'},
            source_manifest='source',candidate_manifest='candidate',step0={'a':[.1,.2]},provenance={}))
    return result


def test_matched_curve_preserves_all_points_regression_and_cost():
    report=module.compare(*arms())
    assert len(report['points'])==11
    assert report['points'][0]['grpo_minus_rloo']['query_macro_set_f1']==0
    assert report['points'][-1]['grpo_minus_rloo']['query_macro_set_f1']==pytest.approx(.05)
    grpo=report['arms']['grpo_v1'];cost=grpo['costs']
    assert grpo['selected_checkpoint']['step']==576
    assert grpo['final_primary_delta']==pytest.approx(.08)
    assert grpo['adjacent_decreases'][0]['step']==634
    assert cost['query_exposures']==81026 and cost['sampled_actions']==1296416
    assert cost['allocated_npu_hours']==16 and cost['summed_slowest_step_seconds']==6340
    assert cost['zero_signal_fraction']==pytest.approx(634/81026)
    assert grpo['final_primary_delta_per_allocated_npu_hour']==pytest.approx(.005)


@pytest.mark.parametrize('key',['config','source_files','source_manifest','candidate_manifest','step0','experiment'])
def test_comparison_rejects_unmatched_runs(key):
    a,b=arms();b[key]='different'
    with pytest.raises(ValueError,match='unmatched'):module.compare(a,b)


@pytest.mark.parametrize('fault',['incomplete','same_container','same_run','different_step0','missing_point','different_initial'])
def test_comparison_rejects_unsupported_curve(fault):
    a,b=arms()
    if fault=='incomplete':b['curve']['complete']=False
    elif fault=='same_container':b['training']['container_id']=a['training']['container_id']
    elif fault=='same_run':b['curve']['run_id']=a['curve']['run_id']
    elif fault=='different_step0':b['curve']['curve'][0]['metrics']['candidate_brier']=.3
    elif fault=='missing_point':b['curve']['observed_steps']=b['curve']['observed_steps'][:-1]
    else:b['curve']['initial_source_policy_id']='other'
    with pytest.raises(ValueError):module.compare(a,b)


def test_infinite_nll_stays_explicit_null():
    a,b=arms();a['curve']['curve'][-1]['metrics']['candidate_binary_nll']=None
    a['curve']['curve'][-1]['binary_nll_infinite']=True
    result=module.compare(a,b)
    assert result['points'][-1]['grpo_minus_rloo']['candidate_binary_nll'] is None


@pytest.mark.parametrize('fault',['missing','exposure','nan','negative_time','zero_signals'])
def test_cost_rejects_invalid_denominators(fault):
    training=arms()[0]['training']
    if fault=='missing':training['updates'].pop()
    elif fault=='exposure':training['updates'][-1]['real_queries']=64
    elif fault=='nan':training['updates'][0]['mean_reward']=float('nan')
    elif fault=='negative_time':training['state']['FinishedAt']='2026-11-14T00:00:00Z'
    else:training['updates'][0]['zero_signal_groups']=129
    with pytest.raises(ValueError):module.cost(training)


def test_paths_normalized_without_masking_different_hyperparameters():
    a=module.normalized({'source':'/run/a/initial','reference':'/run/a/initial','microbatch':8},'/run/a')
    b=module.normalized({'source':'/run/b/initial','reference':'/run/b/initial','microbatch':8},'/run/b')
    assert a==b
    b['microbatch']=16
    assert a!=b


@pytest.fixture
def collected(tmp_path,monkeypatch):
    root=tmp_path/'experiment';root.mkdir();package=tmp_path/'package';package.mkdir()
    collection=tmp_path/'collected';collection.mkdir();point=tmp_path/'point/evaluation/selection';point.mkdir(parents=True)
    def write(path,obj):path.write_text(json.dumps(obj))
    write(root/'status.json',dict(phase='complete_training_and_selection_curve'))
    write(root/'input.json',dict(package=str(package),manifest_sha256='manifest'))
    write(point/'predictions.json',dict(records=[dict(id='q',prob=[.2,.8])]))
    training=dict(algorithm='grpo_v1',source_policy_id='stage1',container_id='actual')
    curve=dict(complete=True,initial_source_policy_id='stage1',independently_audited_points=[
        dict(collection=str(point.parents[1]),predictions_sha256=module.sha(point/'predictions.json'))])
    write(root/'training-audit.json',training);write(root/'curve.json',curve)
    audit_spec=dict(collection=str(collection))
    for key,value in [('transport',{}),('launch',dict(container_id='actual')),('experiment',{'frozen':True})]:
        path=root/(key+'.json');write(path,value);audit_spec[key]=str(path)
    write(root/'training-audit-spec.json',audit_spec)
    write(root/'curve-spec.json',dict(rl_root=str(collection),rl_proof=str(root/'training-audit.json'),
        experiment=audit_spec['experiment'],algorithm='grpo_v1'))
    config=dict(algorithm='grpo_v1',microbatch=8)
    write(collection/'config.json',config);write(package/'config.json',config)
    write(package/'RELEASE.json',dict(source_package_manifest_sha256='source',candidate_manifest_sha256='candidate'))
    calls=[]
    monkeypatch.setattr(module,'verify_package',lambda *args:dict(algorithm='grpo_v1',remote_root='/remote',files={'source/a':'sha'}))
    def train(**args):calls.append('training');return copy.deepcopy(training)
    def evaluate(path):calls.append('curve');return copy.deepcopy(curve)
    monkeypatch.setattr(module,'audit_training',train);monkeypatch.setattr(module,'audit_curve',evaluate)
    return root,calls,point


def test_load_reaudits_training_and_curve(collected):
    root,calls,_=collected
    result=module.load_run(root,'grpo_v1')
    assert calls==['training','curve'] and result['step0']=={'q':[.2,.8]}


@pytest.mark.parametrize('fault',['training','curve','config','predictions','source'])
def test_persisted_success_cannot_override_reaudit(collected,fault):
    root,calls,point=collected
    if fault in ('training','curve'):
        path=root/('training-audit.json' if fault=='training' else 'curve.json');path.write_text('{}')
    elif fault=='config':(root.parent/'collected/config.json').write_text('{}')
    elif fault=='predictions':(point/'predictions.json').write_text('{}')
    else:
        path=root/'curve-spec.json';spec=json.loads(path.read_text());spec['rl_root']='other';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):module.load_run(root,'grpo_v1')
