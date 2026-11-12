from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import compare_intent_binding_update as module
from multimodal_jev.baseline import atomic_json
from test_intent_binding_metrics import fixture as metrics_fixture


def fixture(tmp_path,monkeypatch):
    training=tmp_path/'train';before=tmp_path/'before';suite=tmp_path/'suite'
    for p in (training,before,suite):p.mkdir()
    cfg=dict(image='image',acceptance_root=str(before),pins={},binding_manifest_sha256='manifest')
    atomic_json(training/'config.json',cfg);(training/'run').mkdir()
    selection=dict(source_ids=['a'],step=0,seed=317)
    atomic_json(training/'run/selection.json',selection)
    atomic_json(suite/'verified.json',dict(adapter_id='new'));atomic_json(suite/'ACCEPTED.json',{})
    for root,aid in ((before,'old'),(suite,'new')):
        (root/'run/hf').mkdir(parents=True);(root/'run/vllm').mkdir()
        atomic_json(root/'inputs.json',{})
        atomic_json(root/'config.json',dict(base_id='base',adapter_id=aid,inputs=str(root/'inputs.json')))
        atomic_json(root/'launch.json',dict(image='image',container_id=aid))
        for name in ('source-manifest.json','run/READY.json','run/terminal.json','run/hf/result.json',
                     'run/hf/terminal.json','run/vllm/result.json','run/vllm/terminal.json'):atomic_json(root/name,{})
    rows,records=metrics_fixture();byid={r['id']:r for r in records}
    events=[]
    monkeypatch.setattr(module,'verify_final',lambda *args:events.append('final'))
    monkeypatch.setattr(module,'completed',lambda cid,image:events.append(cid) or True)
    def authenticate(c,policy):
        events.append(policy)
        return {'Q':rows},[byid,byid],dict(passed=True)
    monkeypatch.setattr(module,'authenticate_artifacts',authenticate)
    monkeypatch.setattr(module,'expected_selection',lambda c:([],selection))
    # D/Q task scoring and full gate behavior have separate exhaustive tests.
    original_score=module.score
    scored=original_score(rows,records,'Q')
    monkeypatch.setattr(module,'score',lambda *args:scored)
    def authenticate_both(c,policy):
        events.append(policy)
        return {'D':rows,'Q':rows},[byid,byid],dict(passed=True)
    monkeypatch.setattr(module,'authenticate_artifacts',authenticate_both)
    return training,suite,events


def test_scores_only_after_actual_acceptance_and_both_policy_checks(tmp_path,monkeypatch):
    training,suite,events=fixture(tmp_path,monkeypatch)
    result=module.run(training,suite)
    assert events==['final','old','base:old','new','base:new']
    assert result['role']=='learning' and result['updates']==1
    assert result['comparison']['D']['groups']['other_learning']['recordings']==1


@pytest.mark.parametrize('defect',['acceptance','live_gate','full_gate','schedule','adapter'])
def test_never_scores_unaccepted_or_mismatched_update(tmp_path,monkeypatch,defect):
    training,suite,events=fixture(tmp_path,monkeypatch)
    def reject(*args):raise ValueError('rejected')
    if defect=='acceptance':monkeypatch.setattr(module,'verify_final',reject)
    elif defect=='live_gate':monkeypatch.setattr(module,'completed',lambda *args:False)
    elif defect=='full_gate':monkeypatch.setattr(module,'authenticate_artifacts',reject)
    elif defect=='schedule':atomic_json(training/'run/selection.json',dict(source_ids=['b'],step=0,seed=317))
    else:atomic_json(suite/'verified.json',dict(adapter_id='foreign'))
    with pytest.raises(ValueError):module.run(training,suite)
