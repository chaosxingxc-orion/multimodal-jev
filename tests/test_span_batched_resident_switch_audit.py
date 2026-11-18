"""Candidate switch execution receipts and independent probability rejection."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
import pytest
pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_batch_training_config import PROFILE
from test_simple_sampled_switch_audit import fixture as probabilities


def fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools/jev'))
    import verify_span_batched_resident_switch as audit
    import verify_slurp_batched_parity as gate_verifier
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    root=tmp_path/'switch'; results,refs=probabilities(); pins={}
    for i in range(2):
        gate=tmp_path/f'gate{i}'
        put(gate/'config.json',dict(model='model',base_id='base',runtime_profile='fixed',
            max_model_len=4096,inputs='inputs',population={'requests':256},adapter_id=['old','new'][i]))
        put(gate/'run/hf/result.json',refs[i])
        path=tmp_path/f'audit{i}.json';put(path,dict(backend_root=str(gate)));pins[str(path)]=file_hash(path)
    def accepted(path):
        i=int(Path(path).stem[-1])
        return dict(updates=[2,4][i],requests=256,population_mode='prefix',files={})
    monkeypatch.setattr(gate_verifier,'verify',accepted)
    put(root/'config.json',dict(old_audit_config=str(tmp_path/'audit0.json'),
        updated_audit_config=str(tmp_path/'audit1.json'),candidate_profile=PROFILE,pins=pins))
    launch=tmp_path/'external-launch.json';observed=tmp_path/'external-observed.json'
    put(launch,dict(container_id='switch',image='fixed',config_sha256=file_hash(root/'config.json')))
    put(observed,dict(container_id='switch',image='fixed',state=dict(Status='exited',ExitCode=0)))
    for path in (root/'terminal.json',root/'run/terminal.json'):put(path,dict(exit_code=0))
    put(root/'run/runtime-profile.json',dict(candidate_profile=PROFILE))
    for i,r in enumerate(results):put(root/'run'/f'policy-{i}.json',r)
    behavior=copy.deepcopy(results[1]);behavior['reference_compared']=False
    for row in behavior['records']:row['parity']=None
    put(root/'run/behavior.json',behavior)
    return audit,root,launch,observed,put


def test_external_execution_receipts_and_complete_recomputation(tmp_path,monkeypatch):
    m,root,launch,observed,_=fixture(tmp_path,monkeypatch)
    r=m.verify(root,observed,launch)
    assert r['passed'] and r['exact_switch_back'] and r['reference_free_behavior']['passed']
    assert r['files'][str(launch)]==file_hash(launch)


@pytest.mark.parametrize('fault',['live','config','profile','partial','probability','behavior'])
def test_reject_invalid_candidate_switch(tmp_path,monkeypatch,fault):
    m,root,launch,observed,put=fixture(tmp_path,monkeypatch)
    p=observed if fault=='live' else root/'config.json' if fault=='config' else root/'run/runtime-profile.json' if fault=='profile' else root/'run/behavior.json' if fault=='behavior' else root/'run/policy-1.json'
    d=json.loads(p.read_text())
    if fault=='live':d['state']['Status']='running'
    elif fault in ('config','profile'):d['candidate_profile']='other'
    elif fault=='partial':d['records'].pop()
    elif fault=='behavior':d['reference_compared']=True
    else:d['records'][0]['candidate_logprobs']=[[-4.0943445622221]*60]
    put(p,d)
    with pytest.raises(ValueError):m.verify(root,observed,launch)
