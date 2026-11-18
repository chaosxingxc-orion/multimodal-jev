from pathlib import Path

import pytest

from tools.reports import context_cached_grpo_transfer_source as m
from test_context_transfer_source_admission import fixture
from test_prepare_context_train_parity_job import write


@pytest.mark.parametrize('fault', [None,'score','policy','contract'])
def test_cached_training_layout_with_independent_curve_gates(tmp_path, monkeypatch, fault):
    c, _, _ = fixture(tmp_path, monkeypatch)
    checked = m.shared.reconstruct(c['source_curve_config'])
    training = tmp_path/'cached-package'
    write(training/'config.json', dict(source_contract={},base_id='base'))
    checked['plan']['source_operations']['training_package']={'output':str(training)}
    monkeypatch.setattr(m,'reconstruct',lambda _:checked)
    if fault=='score':checked['points'][0]['result']['metrics']['candidate_brier']=.999
    elif fault=='policy':c['policy_id']='wrong'
    elif fault=='contract':
        def refuse(*a,**kw):raise ValueError('real policy tensors differ')
        monkeypatch.setattr(m.shared,'read_execution_bundle',refuse)
    if fault:
        with pytest.raises(ValueError):m.admit(c)
    else:
        result=m.admit(c)
        assert result['binding']['step']==128 and not result['target_inputs_opened']
        root=tmp_path/'portable-source';entry=m.seal(result,root,'/remote/source')
        assert m.verify_bundle(root,'/remote/source',result)==entry
