from copy import deepcopy
import json

import pytest

from multimodal_jev.sfc_prompt_selection import compare, population_scores
from test_sfc_teacher_feedback import fixture
from test_sfc_jev_runtime import predictions


def data():
    rows,gold,_=fixture()
    return rows,gold,[p for r,g in zip(rows,gold) for p in predictions(r,g)]


def test_exact_tie_keeps_parent_and_reports_activity():
    rows,gold,ps=data()
    report=compare(rows,gold,{'parent':ps,'p1':deepcopy(ps)},['parent','p1'])
    assert report['winner']=='parent'
    assert report['scores']['parent']['counts']['complete_conversations']==2
    assert sum(v['functions'] for v in report['scores']['parent']['activity'].values())==8
    assert report['paired']['p1']['complete_conversations']['both_correct']==2


def test_fixing_last_bad_decision_completes_conversation():
    rows,gold,ps=data();parent=deepcopy(ps);parent[0]['raw_text']='invalid'
    report=compare(rows,gold,{'parent':parent,'p1':ps},['parent','p1'])
    assert report['winner']=='p1'
    assert report['paired']['p1']['complete_conversations']['fixed']==1
    assert report['paired']['p1']['complete_conversations']['regressed']==0


def test_calls_regression_disqualifies_more_complete_conversations():
    rows,gold,ps=data();parent=deepcopy(ps);candidate=deepcopy(ps)
    # Parent has a wrong Noul in each conversation but all native calls correct.
    for uid in (rows[0]['id'],rows[2]['id']):
        next(p for p in parent if p['example_id']==uid and p['decision_id'].endswith('.present'))['raw_text']='invalid'
    # Candidate fixes both Noul errors but breaks one calls result.
    next(p for p in candidate if p['example_id']==rows[2]['id'] and p['decision_id'].endswith('.calls'))['raw_text']='invalid'
    report=compare(rows,gold,{'parent':parent,'p1':candidate},['parent','p1'])
    assert report['scores']['p1']['counts']['complete_conversations']==1
    assert report['scores']['parent']['counts']['complete_conversations']==0
    assert report['winner']=='parent' and 'native_calls_exact' in report['ineligible_reasons']['p1']


def test_multiple_candidates_tie_uses_registered_order():
    rows,gold,ps=data();parent=deepcopy(ps);parent[0]['raw_text']='invalid'
    report=compare(rows,gold,{'p2':ps,'parent':parent,'p1':deepcopy(ps)},['parent','p1','p2'])
    assert report['winner']=='p1'


@pytest.mark.parametrize('change',['missing_candidate','missing_decision','duplicate','unknown','test','turn','roster','targets'])
def test_incomplete_population_never_selects_a_prompt(change):
    rows,gold,ps=data();preds={'parent':ps,'p1':deepcopy(ps)};roster=['parent','p1']
    if change=='missing_candidate':preds.pop('p1')
    if change=='missing_decision':preds['p1'].pop()
    if change=='duplicate':preds['p1'].append(deepcopy(ps[0]))
    if change=='unknown':preds['p1'][0]['example_id']='other'
    if change=='test':rows[0]['split']='test_id'
    if change=='turn':rows.pop();gold.pop()
    if change=='roster':roster.reverse()
    if change=='targets':gold.pop()
    with pytest.raises(ValueError):compare(rows,gold,preds,roster)


@pytest.mark.parametrize('change',[None,'failed_candidate','runtime','instruction','learning','rules','dispatch'])
def test_comparison_checks_all_runs_before_selection_labels(tmp_path,monkeypatch,change):
    from pathlib import Path
    from multimodal_jev.baseline import file_hash
    from multimodal_jev.consumption import canonical
    from test_sfc_jev_merge import fixture as merge_fixture
    from test_sfc_jev import load_tool
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools/jev'))
    merger,plan,roster,release,_=merge_fixture(tmp_path,monkeypatch,'learning' if change=='learning' else 'selection')
    baseline=merger.collect(plan,file_hash(plan),roster,file_hash(roster))
    tool=load_tool('compare_sfc_prompt_selection');calls=[]
    def collect(*args):
        result=deepcopy(baseline);actor=result[-1];name='parent' if not calls else 'p1';calls.append(name)
        if change=='failed_candidate' and name=='p1':raise ValueError('candidate incomplete')
        actor['config'].update(prompt_id=name,instruction_sha256=name+'hash')
        actor['config']['candidate']=None if name=='parent' else {'fixture':True}
        if name=='p1' and change=='runtime':actor['config']['seed']=99
        if name=='p1' and change=='instruction':actor['config']['instruction_sha256']='other'
        return result
    monkeypatch.setattr(tool,'collect',collect)
    spec=dict(schema='sfc-jev-prompt-comparison-v1',plan=str(plan),plan_sha256=file_hash(plan),
        rules_files=tool.rules_files(),roster=['parent','p1'],prompts={name:dict(roster_path=str(roster),instruction_sha256=name+'hash',
            candidate=None if name=='parent' else {'fixture':True}) for name in ('parent','p1')})
    if change=='rules':spec['rules_files']['src/multimodal_jev/sfc_jev_score.py']='changed'
    path=tmp_path/'spec.json';path.write_bytes(canonical(spec))
    queue=roster.parent/'queue-plan.json'
    queue.write_bytes(canonical(dict(comparison=dict(path=str(path),sha256=file_hash(path)))))
    (roster.parent/'dispatch-intent.json').write_bytes(canonical(dict(
        queue_plan_sha256=file_hash(queue),comparison_sha256='wrong' if change=='dispatch' else file_hash(path))))
    if change is None:
        result=tool.run(path,file_hash(path))
        assert calls==['parent','p1'] and result['winner']=='parent'
        assert result['scores']['parent']['counts']['bundles']==2
    else:
        # No target file: the expected engineering failure must occur first.
        (release/'train_id.targets.jsonl').unlink()
        with pytest.raises(ValueError):tool.run(path,file_hash(path))


@pytest.mark.parametrize('change',[None,'missing','rules','prompt','root','plan','nonselection'])
def test_selection_queue_requires_matching_preregistered_spec(tmp_path,monkeypatch,change):
    from pathlib import Path
    from multimodal_jev.baseline import file_hash,atomic_json
    from test_sfc_jev_merge import fixture as merge_fixture
    from test_sfc_jev import load_tool
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools/jev'))
    _,plan_path,_,_,_=merge_fixture(tmp_path,monkeypatch,'selection')
    queue=load_tool('run_sfc_jev_shard_queue')
    comparator=load_tool('compare_sfc_prompt_selection')
    plan=json.loads(plan_path.read_text());root=tmp_path/'new-queue'
    cfg=dict(shard_plan=str(plan_path),shard_plan_sha256=file_hash(plan_path),
        prompt_id='parent',instruction_sha256='parenthash')
    spec=dict(schema='sfc-jev-prompt-comparison-v1',plan=str(plan_path),plan_sha256=file_hash(plan_path),
        rules_files=comparator.rules_files(),roster=['parent','p1'],prompts={
            p:dict(roster_path=str((root if p=='parent' else tmp_path/'p1')/'completed-runs.json'),
                   instruction_sha256=p+'hash',candidate=None) for p in ('parent','p1')})
    if change=='rules':spec['rules_files']={}
    path=tmp_path/'comparison.json';atomic_json(path,spec)
    binding=dict(path=str(path),sha256=file_hash(path))
    if change=='missing':binding=None
    if change=='prompt':cfg['instruction_sha256']='changed'
    if change=='root':root=tmp_path/'other'
    if change=='plan':plan={**plan,'bundles':0}
    if change=='nonselection':plan={**plan,'partition':None}
    if change is None:assert queue.comparison_binding(root,cfg,plan,binding)==binding
    else:
        with pytest.raises(ValueError):queue.comparison_binding(root,cfg,plan,binding)
    assert not root.exists()
