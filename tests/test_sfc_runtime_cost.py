import copy
from pathlib import Path
import sys

import pytest

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.sfc_jev_shards import make_plan,population
from multimodal_jev.sfc_runtime_cost import project
from test_sfc_jev_shards import fixture as input_fixture
from test_sfc_jev import load_tool

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def fixture(tmp_path):
    release,_=input_fixture(tmp_path);plan=make_plan(release,'test_id',12)
    rows=population(release,'test_id')
    predictions=[dict(example_id=r['id'],decision_id=d['id'],
        seconds=len(r['audio_sha256'])*(3 if d['id'].endswith('.calls') else 1))
        for r in rows[:2] for d in r['input']['decisions']]
    elapsed=sum(p['seconds'] for p in predictions)+10
    return rows,plan,predictions,elapsed


def test_complete_projection_counts_each_decision_and_container_overhead(tmp_path):
    rows,plan,ps,elapsed=fixture(tmp_path)
    report=project(rows[:2],ps,elapsed,rows,plan)
    assert report['observed_decisions']==12 and report['population_decisions']==24
    assert len(report['stratum_measurements'])==6 and report['shard_count']==2
    assert report['observed_residual_seconds_per_container']==10
    # Two identical source-group workloads, each pays the observed residual once.
    assert report['estimated_serial_wall_days']==pytest.approx(2*elapsed/86400)
    assert report['estimated_npu_device_hours']==pytest.approx(2*elapsed*2/3600)
    assert report['dispatched'] is False


@pytest.mark.parametrize('bad',[True,-1,float('nan'),float('inf'),'1',None])
def test_invalid_measurements_rejected(tmp_path,bad):
    rows,plan,ps,elapsed=fixture(tmp_path);ps[0]['seconds']=bad
    with pytest.raises(ValueError,match='finite'):project(rows[:2],ps,elapsed,rows,plan)


@pytest.mark.parametrize('change',['missing','duplicate','unknown','short_elapsed','missing_bundle','duplicate_bundle','plan_count'])
def test_incomplete_evidence_cannot_produce_estimate(tmp_path,change):
    rows,plan,ps,elapsed=fixture(tmp_path)
    observed=copy.deepcopy(rows[:2])
    if change=='missing':ps.pop()
    if change=='duplicate':ps.append(ps[0])
    if change=='unknown':ps[0]['decision_id']='unknown'
    if change=='short_elapsed':elapsed=0
    if change=='missing_bundle':rows.pop()
    if change=='duplicate_bundle':rows.append(rows[0])
    if change=='plan_count':plan['decisions']-=1
    with pytest.raises(ValueError):project(observed,ps,elapsed,rows,plan)


def test_unobserved_two_audio_stratum_has_no_mean_fallback(tmp_path):
    rows,plan,ps,elapsed=fixture(tmp_path)
    ps=[p for p in ps if p['example_id']==rows[0]['id']]
    with pytest.raises(ValueError,match='unobserved'):project(rows[:1],ps,elapsed,rows,plan)


def test_observed_range_is_sensitivity_and_flags_timeout_without_shrinking(tmp_path):
    rows,plan,ps,elapsed=fixture(tmp_path)
    for p in ps:p['seconds']=10000 if p['decision_id'].startswith('f00') else 20000
    report=project(rows[:2],ps,sum(p['seconds'] for p in ps)+10,rows,plan)
    assert report['observed_min_sensitivity_wall_days']<report['estimated_serial_wall_days']<report['observed_max_sensitivity_wall_days']
    assert all(s['mean_exceeds_86400_second_container_limit'] for s in report['shards'])
    assert report['population_decisions']==24 and report['shard_count']==2
    assert 'not confidence intervals' in report['limitations']


def test_tool_requires_actual_acceptance_and_pins_complete_population(tmp_path,monkeypatch):
    m=load_tool('estimate_sfc_jev_cost');rows,plan,ps,elapsed=fixture(tmp_path)
    path=tmp_path/'plan.json';atomic_json(path,plan)
    root=tmp_path/'probe';(root/'run').mkdir(parents=True)
    atomic_json(root/'config.json',dict(versions={'vllm':'pin'},audio_model='/model',max_new_tokens=1024))
    atomic_json(root/'launch.json',dict(container_id='exact-cid',image='image'))
    atomic_json(root/'acceptance.json',{})
    atomic_json(root/'run/result.json',dict(elapsed_seconds=elapsed))
    (root/'run/predictions.jsonl').write_text('synthetic binding fixture')
    events=[]
    monkeypatch.setattr(m,'check_probe',lambda _:events.append('check'))
    monkeypatch.setattr(m,'accept',lambda _:events.append('accept') or
        (rows[:2],dict(release=plan['release'],manifest_sha256=plan['manifest_sha256']),ps))
    result=m.estimate(root,path,file_hash(path))
    assert events==['check','accept'] and result['probe_container_id']=='exact-cid'
    assert result['population_decisions']==24 and result['plan_sha256']==file_hash(path)
    assert result['targets_read_by_estimator'] is False
    def failed(_):raise ValueError('actual container failed')
    monkeypatch.setattr(m,'accept',failed)
    with pytest.raises(ValueError,match='actual container'):m.estimate(root,path,file_hash(path))
