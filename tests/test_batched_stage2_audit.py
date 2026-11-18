"""Candidate boundary audit checks saved execution metadata and exact masters."""
import json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev import batched_stage2_audit as m
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_batch_training_config import PROFILE,batch_layout
from test_simple_stage2_audit import fixture as base_fixture


def fixture(tmp_path,monkeypatch):
    c,root,put=base_fixture(tmp_path,monkeypatch)
    import multimodal_jev.simple_stage2_audit as old
    monkeypatch.setattr(m,'authenticate',old.authenticate)
    p=root/'config.json';v=json.loads(p.read_text());v.update(candidate_profile=PROFILE,
        global_batch=128,microbatch_per_rank=8,batch_order='length_grouped');put(p,v)
    layout=batch_layout(v)
    p=root/'launch.json';v=json.loads(p.read_text());v['config_sha256']=file_hash(root/'config.json')
    launch=tmp_path/'external-launch.json';put(launch,v);p.unlink()
    c['training_launch']=str(launch);c['pins'].pop(str(p))
    for p in (root/'config.json',launch):c['pins'][str(p)]=file_hash(p)
    for rank in range(4):
        p=root/'run'/f'rank-{rank}/stage2-initial.pt';state=torch.load(p,weights_only=True)
        state.update(candidate_profile=PROFILE,batch_layout=layout,batch_order='length_grouped');torch.save(state,p)
    return c,root


def test_four_rank_candidate_boundary(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch);r=m.source(c,verify_parity=None)
    assert len(r['checks'])==4 and all(v['exact_masters'] for v in r['checks'])


@pytest.mark.parametrize('fault',['profile','micro','order','master','adam'])
def test_candidate_boundary_rejects_changed_saved_state(tmp_path,monkeypatch,fault):
    c,root=fixture(tmp_path,monkeypatch);p=root/'run/rank-2/stage2-initial.pt';s=torch.load(p,weights_only=True)
    if fault=='profile':s['candidate_profile']='other'
    elif fault=='micro':s['batch_layout']['microbatch_per_rank']=16
    elif fault=='order':s['batch_order']='original'
    elif fault=='master':s['parameters'][s['names'][0]]+=.01
    else:s['optimizer']['state']={0:{'step':torch.tensor(1.)}}
    torch.save(s,p)
    with pytest.raises(ValueError):m.source(c,verify_parity=None)
