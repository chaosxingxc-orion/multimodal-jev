from pathlib import Path
import copy

import pytest

from test_context_transfer_macro_parity import reports
from tools.reports import context_full_transfer_operations as m


@pytest.mark.parametrize('fault',[None,'hf','panel','service','subset'])
def test_full_outputs_compare_actual_hf_subset_not_saved_parity_flag(tmp_path,monkeypatch,fault):
    selected,hf,records=reports();selected['indices']=list(range(len(records)))
    p=tmp_path/'package';r=tmp_path/'collection'
    e=dict(snapshot='/remote/package/policy',dataset=selected['dataset'],policy_id='p',contract={},
        policy_role='common_Stage1',portable_full_inputs={},
        macro_client=dict(attempt_id='full',runtime_identity={'profile':'fixture'}))
    def read(path):
        path=Path(path);name=path.name
        if name=='evaluation.json':return e
        if name=='hf.json':return dict(snapshot=e['snapshot'],portable_parity_inputs={})
        if name=='terminal.json':return dict(exit_code=0)
        if name=='panel.json':return dict(selected,dataset='other') if fault=='panel' else selected
        if name=='hf-runtime.json':
            from multimodal_jev.context_runtime_profile import PROFILE
            return {'profile':PROFILE}
        if name=='ready.json':return dict(physical_devices=[6 if path.parent.name=='service-0' else 7],
            policy_id='wrong' if fault=='service' else 'p',contract={},max_batch_queries=32,model_forward_mode='fixture')
        if name=='predictions.json':
            data=copy.deepcopy(records)
            if fault=='subset':data.reverse()
            return {'records':data}
        if name=='reference.json':
            value=copy.deepcopy(hf)
            if fault=='hf':value['policy_id']='wrong'
            # reports() binds the prior panel; indices are irrelevant to comparison
            from multimodal_jev.context_macro_rollout import digest
            value['panel_sha256']=digest(selected)
            return value
        raise AssertionError(path)
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(m,'load_panel',lambda _:dict(selected_panel=selected))
    monkeypatch.setattr(m,'coverage',lambda *a:None)
    monkeypatch.setattr(m,'audit_full',lambda *a:dict(recordings=len(records)))
    if fault:
        with pytest.raises(ValueError):m.audit_outputs(p,r)
    else:
        result=m.audit_outputs(p,r)
        assert result['parity']['exact'] and result['target_scored'] is False
        assert result['runtime_provenance_verified'] is False
