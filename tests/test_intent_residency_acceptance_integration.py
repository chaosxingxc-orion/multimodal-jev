"""Exercise frozen acceptance wrapper failure paths and real tensor comparison.

Gate receipts are mocks here. Actual Docker and audio/NPU gates remain required.
"""
import copy
from pathlib import Path
import runpy
import sys
import types
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
import verify_intent_residency_replay_cpu as module


def fixture(tmp_path,monkeypatch):
    reference=tmp_path/'reference';training=tmp_path/'training';suite=tmp_path/'suite';ref_suite=tmp_path/'reference-suite'
    for root in (reference,training,suite,ref_suite):root.mkdir()
    atomic_json(training/'config.json',dict(pins={},replay_reference=str(reference),replay_reference_suite=str(ref_suite)))
    for root in (training,reference):
        (root/'run').mkdir()
        for name in ('selection.json','supervision.json'):atomic_json(root/'run'/name,{'same':True})
        for rank in range(4):
            out=root/'run'/f'rank-{rank}';out.mkdir()
            p=torch.tensor([.1234567],dtype=torch.float32)
            state=dict(rank=rank,world_size=4,updates=2,names=['w'],parameters={'w':p},
                optimizer=dict(param_groups=[{'params':[0]}],state={0:dict(step=torch.tensor(2.),exp_avg=p/10,exp_avg_sq=p/100)}))
            torch.save(state,out/'optimizer-local.pt')
            atomic_json(out/'residency.json',dict(rank=rank,passed=True,exact_parameters_and_adam=True,
                original_tensor_devices_preserved=True,parameter_optimizer_references_preserved=True))
    for root,gate in ((reference,ref_suite),(training,suite)):
        atomic_json(gate/'verified.json',dict(root=str(root),updates=2,arm='D',files={}))
    monkeypatch.setattr(module,'validate',lambda c:None)
    monkeypatch.setattr(module,'verify_final',lambda gate:dict(adapter_id='same'))
    monkeypatch.setattr(module,'checkpoint_identity',lambda path:('same',{}))
    return reference,training,suite


def test_full_real_tensor_comparison_under_mock_gate(tmp_path,monkeypatch):
    _,training,suite=fixture(tmp_path,monkeypatch);r=module.verify(training,suite)
    assert r['passed'] and [x['rank'] for x in r['ranks']]==list(range(4))
    assert sum(x['tensors_compared'] for x in r['ranks'])==16


@pytest.mark.parametrize('defect',['master','adam','rank','residency','selection','export'])
def test_gate_success_cannot_hide_replay_defects(tmp_path,monkeypatch,defect):
    reference,training,suite=fixture(tmp_path,monkeypatch)
    path=training/'run/rank-3/optimizer-local.pt'
    if defect in ('master','adam','rank'):
        s=torch.load(path,weights_only=True)
        if defect=='master':s['parameters']['w'].add_(1e-7)
        elif defect=='adam':s['optimizer']['state'][0]['exp_avg'].add_(1e-7)
        else:s['rank']=2
        torch.save(s,path)
    elif defect=='residency':atomic_json(training/'run/rank-3/residency.json',dict(rank=3,passed=False))
    elif defect=='selection':atomic_json(training/'run/selection.json',{'same':False})
    else:monkeypatch.setattr(module,'checkpoint_identity',lambda path:('different' if str(path).startswith(str(training)) else 'same',{}))
    with pytest.raises(ValueError):module.verify(training,suite)


@pytest.mark.parametrize('failure',[None,'ordinary','residency'])
def test_wrapper_never_publishes_replay_acceptance_on_failure(tmp_path,monkeypatch,failure):
    # Execute the actual __main__ wrapper, with only expensive numerical phases mocked.
    def ordinary(root):
        if failure=='ordinary':raise ValueError('injected ordinary failure')
        atomic_json(root/'ACCEPTED.json',dict(passed=True))
    def replay(training,root):
        if failure=='residency':raise ValueError('injected replay failure')
        return dict(passed=True,scope='mock-control')
    monkeypatch.setitem(sys.modules,'run_intent_finite_acceptance',types.SimpleNamespace(run=ordinary))
    monkeypatch.setitem(sys.modules,'verify_intent_residency_replay_cpu',types.SimpleNamespace(verify=replay))
    atomic_json(tmp_path/'config.json',dict(training_root='/mock'))
    monkeypatch.setattr(sys,'argv',['wrapper','--root',str(tmp_path)])
    script=Path(__file__).resolve().parents[1]/'tools/jev/run_intent_residency_acceptance.py'
    if failure:
        with pytest.raises(ValueError):runpy.run_path(str(script),run_name='__main__')
        assert not (tmp_path/'RESIDENCY_ACCEPTED.json').exists()
        assert module.read(tmp_path/'residency-terminal.json')['exit_code']==1
    else:
        runpy.run_path(str(script),run_name='__main__')
        assert module.read(tmp_path/'RESIDENCY_ACCEPTED.json')['verified_sha256']==file_hash(tmp_path/'residency-verified.json')
        assert module.read(tmp_path/'residency-terminal.json')['exit_code']==0
