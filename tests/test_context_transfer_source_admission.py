import json
from pathlib import Path

import pytest

from test_collect_context_macro_source_curve import setup
from tools.reports import collect_context_macro_source_curve as curve
from tools.reports import context_transfer_source_admission as m
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.evaluate_context_transfer_macro import source_binding


def fixture(tmp_path,monkeypatch):
    c,plan,ops,audits,calls=setup(tmp_path,monkeypatch)
    result=curve.collect(c,{})
    parent=tmp_path/'training';(parent/'package').mkdir(parents=True)
    (parent/'package/training.json').write_text(json.dumps(dict(source_contract={},base_id='base')))
    plan['source_operations']['prepared']=str(parent)
    for point in plan['checkpoints']:
        point.update(snapshot=str(Path(ops[str(point['step'])]['prepared'])/'package/policy'),adapter='/adapter')
    # The fixture re-auditor provides a final plan; match stored point plans after setup.
    for item in ops.values():
        (Path(item['prepared'])/'package/SOURCE-PLAN.json').write_text(json.dumps(plan))
    p=Path(c['source_plan']);p.write_text(json.dumps(plan));c['source_plan_sha256']=curve.file_hash(p)
    # Re-publish into a new directory after final fixture identities are bound.
    c['output']=str(tmp_path/'final-curve');result=curve.collect(c,{})
    monkeypatch.setattr(m,'read_execution_bundle',lambda *a,**kw:kw)
    monkeypatch.setattr(m,'verify_native_adapter',lambda *a:None)
    spec=dict(source_evaluation=result['source_evaluation'],source_curve_config=c,
              policy_role='source_selected_new_GRPO',policy_id='p128')
    return spec,audits,calls


def test_actual_curve_reconstructed_and_portable_scores_survive_move(tmp_path,monkeypatch):
    c,audits,calls=fixture(tmp_path,monkeypatch);calls.clear()
    admitted=m.admit(c)
    assert calls==['training',*[str(s) for s in [0,*range(64,634,64),634]]]
    assert admitted['binding']['step']==128 and admitted['target_inputs_opened'] is False
    root=tmp_path/'bundle';entry=m.seal(admitted,root,'/remote/source')
    assert m.verify_bundle(root,'/remote/source',admitted)==entry
    source=root/'source-evaluation.json'
    content=relocate(json.loads(source.read_text()),'/remote/source',str(root))
    source.write_text(json.dumps(content))
    checked=source_binding(dict(source_evaluation=dict(path=str(source),sha256=m.file_hash(source)),
                               policy_role=c['policy_role'],policy_id=c['policy_id']))
    assert checked['step']==128


@pytest.mark.parametrize('fault',['score','runtime','weights','chosen','audit'])
def test_stale_published_source_never_admits_target_policy(tmp_path,monkeypatch,fault):
    c,audits,_=fixture(tmp_path,monkeypatch)
    if fault=='score':audits['128']['result']['metrics']['candidate_brier']=.1
    if fault=='runtime':audits['64']['image']='other'
    if fault=='weights':
        def fail(*a,**kw):raise ValueError('weight mismatch')
        monkeypatch.setattr(m,'read_execution_bundle',fail)
    if fault=='chosen':c['policy_id']='p64'
    if fault=='audit':
        p=Path(c['source_evaluation']['path']);v=json.loads(p.read_text())
        v['source_audits']['sha256']='wrong';p.write_text(json.dumps(v));c['source_evaluation']['sha256']=m.file_hash(p)
    with pytest.raises(ValueError):m.admit(c)


@pytest.mark.parametrize('fault',['score','origin','extra','symlink'])
def test_resealed_source_bundle_rejected(tmp_path,monkeypatch,fault):
    c,_,_=fixture(tmp_path,monkeypatch);a=m.admit(c);root=tmp_path/'bundle';m.seal(a,root,'/remote/source')
    if fault=='score':(root/'score-0128.json').write_text('{}')
    if fault=='origin':(root/'ORIGIN.json').write_text('{}')
    if fault=='extra':(root/'target-score.json').write_text('{}')
    if fault=='symlink':
        p=root/'curve.json';p.rename(tmp_path/'outside');p.symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError):m.verify_bundle(root,'/remote/source',a)
