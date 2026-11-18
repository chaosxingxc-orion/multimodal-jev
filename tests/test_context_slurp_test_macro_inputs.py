import pytest

from tools.reports import context_slurp_test_macro_inputs as module
from tools.reports.context_transfer_macro_inputs import normalize
from test_context_transfer_macro_inputs import fixture


def source_fixture(monkeypatch):
    rows=[dict(id=f'test{i}',audio_sha256=f'test-audio{i}') for i in range(12991)]
    task=dict(schema=dict(candidates=[dict(id=str(i),description=f'candidate {i}') for i in range(60)]),
        thinking=dict(requirements=['listen']))
    definitions=dict(candidates=[dict(id=c['id'],definition=c['description']) for c in task['schema']['candidates']],
        requirements=['listen'],semantics='exclusive_event')
    source={role:[dict(id=f'{role}{i}',audio_sha256=f'{role}-audio{i}') for i in range(count)]
        for role,count in [('learning',40513),('selection',9226)]}
    calls=[]
    def load(path,digest,role,*,purpose):
        assert path=='release' and digest=='hash'
        assert purpose==('optimize' if role=='learning' else 'evaluate')
        calls.append(role)
        return source[role],task,dict(counts={k:dict(recordings=len(v)) for k,v in source.items()})
    monkeypatch.setattr(module,'load_release',load)
    return rows,definitions,source,calls


def test_full_test_binds_current_both_source_roles_and_task(monkeypatch):
    rows,definitions,_,calls=source_fixture(monkeypatch)
    module.bind_source(rows,definitions,dict(path='release',sha256='hash'))
    assert calls==['learning','selection']


@pytest.mark.parametrize('fault',['id_learning','audio_selection','definition','requirement','source_missing','test_missing'])
def test_overlap_or_changed_task_or_population_rejected(monkeypatch,fault):
    rows,definitions,source,_=source_fixture(monkeypatch)
    if fault=='id_learning':source['learning'][0]['id']=rows[0]['id']
    if fault=='audio_selection':source['selection'][0]['audio_sha256']=rows[0]['audio_sha256']
    if fault=='definition':definitions['candidates'][0]['definition']='changed'
    if fault=='requirement':definitions['requirements'].append('changed')
    if fault=='source_missing':source['learning'].pop()
    if fault=='test_missing':rows.pop()
    with pytest.raises(ValueError):module.bind_source(rows,definitions,dict(path='release',sha256='hash'))


def test_60_candidate_input_normalization_preserves_receipts():
    rows,receipts,requests,ids=fixture(60)
    for row in rows:row['source_group']='group'
    population,normalized=normalize(rows,receipts,requests,ids,expected_count=3,decisions=60)
    assert len(normalized)==3 and len(normalized[0]['positions'])==60
    assert all(set(r)=={'id','audio_sha256','request_sha256'} for r in population)


def test_test_inputs_reject_source_curve_release_mismatch_before_opening_data(tmp_path,monkeypatch):
    from test_context_transfer_macro_evaluation import source_config
    from tools.reports import evaluate_context_transfer_macro as entry
    c=source_config(tmp_path)
    c.update(dataset='SLURP_clean_TEST60',source_release=dict(path='release',sha256='other'))
    def forbidden(*a,**k):pytest.fail('must not open TEST before source binding')
    monkeypatch.setattr(entry,'load',forbidden)
    with pytest.raises(ValueError,match='source release'):entry.run(c,{})
