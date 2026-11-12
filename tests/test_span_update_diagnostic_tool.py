import importlib.util
import json
from pathlib import Path
import pytest


def test_dataset_ids_are_resolved_to_authenticated_runtime_request_ids(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'tools/jev/analyze_span_transfer_updates.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('update_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'verify',lambda root:dict(initial_completed_updates=0,completed_updates=1))
    monkeypatch.setattr(module,'step_rows',lambda config:[dict(id='dataset-row',targets=[True],supervised=[True])])
    stage=tmp_path/'run/update-0001';(stage/'run').mkdir(parents=True)
    runtime=tmp_path/'runtime';(runtime/'run').mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    write(tmp_path/'config.json',dict(method='conditional',seed=17))
    write(stage/'config.json',dict(runtime_admission=str(runtime),auxiliary_weight=0.))
    write(runtime/'run/result.json',dict(records=[dict(id='dataset-row',request_id='bound-input-policy-hash')]))
    policy=[[[-0.6931471805599453,-0.6931471805599453]]]
    write(stage/'run/before.json',dict(request_ids=['bound-input-policy-hash'],candidate_logprobs=policy))
    write(stage/'run/after.json',dict(candidate_logprobs=policy))
    write(stage/'run/update.json',dict(gradient_norm=1.,loss=0.))
    result=module.analyze(tmp_path)
    assert result['updates'][0]['reward_delta']==0
    assert str(runtime/'run/result.json') in result['files']
    write(stage/'run/before.json',dict(request_ids=['dataset-row'],candidate_logprobs=policy))
    with pytest.raises(ValueError,match='request order'):
        module.analyze(tmp_path)
