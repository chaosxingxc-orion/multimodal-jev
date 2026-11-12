import copy
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import verify_intent_resident_replay_cpu as module
from multimodal_jev.baseline import atomic_json,file_hash
from test_intent_resident_replay import state


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'resident';suite=tmp_path/'suite';refs=[tmp_path/'ref1',tmp_path/'ref2'];gates=[tmp_path/'gate1',tmp_path/'gate2']
    for p in [root,suite,*refs,*gates]:(p/'run').mkdir(parents=True)
    plan=dict(references=[dict(training=str(t),suite=str(s)) for t,s in zip(refs,gates)])
    atomic_json(root/'config.json',plan);atomic_json(root/'source-manifest.json',{})
    atomic_json(root/'launch.json',dict(config_sha256=file_hash(root/'config.json'),source_manifest_sha256=file_hash(root/'source-manifest.json')))
    cfg=dict(resident_training_root=str(root),adapter=str(root/'run/resident-adapter'),adapter_id='a2',base_id='base',population={'recordings':89,'requests':178},pins={})
    atomic_json(suite/'config.json',cfg);atomic_json(suite/'source-manifest.json',{});atomic_json(suite/'launch.json',{})
    for t,g in zip(refs,gates):atomic_json(g/'verified.json',dict(root=str(t),files={}))
    atomic_json(gates[1]/'config.json',dict(base_id='base'))
    for name in ('selection.json','supervision.json','resident-selection.json'):atomic_json(root/'run'/name,{'same':True})
    for t in refs:
        for name in ('selection.json','supervision.json'):atomic_json(t/'run'/name,{'same':True})
    for rank in range(4):
        for step in (1,2):
            folder=root/'run'/(f'rank-{rank}' if step==1 else f'resident-rank-{rank}');folder.mkdir()
            ref=refs[step-1]/'run'/f'rank-{rank}';ref.mkdir()
            s=state(step);s['rank']=rank
            for p in (folder,ref):
                torch.save(s,p/'optimizer-local.pt')
                for name in ('before.json','after.json'):atomic_json(p/name,{'same':True})
            if step==1:atomic_json(folder/'terminal.json',dict(exit_code=0))
            else:atomic_json(folder/'result.json',dict(passed=True,rank=rank,updates=2,phase='awaiting_acceptance',resident_objects_preserved=True,policy_id='base:a2'))
    records=[dict(id=str(i),request_id=str(i),request_identity={'same':True},candidate_logprobs=[[-.6931471805599453]*2]) for i in range(178)]
    for g in (suite,gates[1]):
        for backend in ('hf','vllm'):
            (g/'run'/backend).mkdir();atomic_json(g/f'run/{backend}/result.json',dict(records=records))
    monkeypatch.setattr(module,'validate',lambda p:[dict(pilot='pilot',manifest_sha256='hash')]*2)
    monkeypatch.setattr(module,'verify_final',lambda p:dict(passed=True))
    monkeypatch.setattr(module,'verify_gate',lambda *a:dict(passed=True,requests=178))
    def identity(path):
        path=Path(path)
        return ('a2' if path==root/'run/resident-adapter' or path==refs[1]/'run/adapter' else 'a1'),{}
    monkeypatch.setattr(module,'checkpoint_identity',identity)
    return root,suite


def test_both_real_tensor_states_and_full_backend_records(tmp_path,monkeypatch):
    r,s=fixture(tmp_path,monkeypatch);report=module.verify(r,s)
    assert report['passed'] and len(report['ranks'])==4 and report['full_gate']['requests']==178


@pytest.mark.parametrize('bad',['master','adam','before','backend','phase','rank','schedule'])
def test_independent_verifier_rejects_hidden_replay_defects(tmp_path,monkeypatch,bad):
    r,s=fixture(tmp_path,monkeypatch);out=r/'run/resident-rank-3'
    if bad in ('master','adam'):
        p=out/'optimizer-local.pt';state=torch.load(p,weights_only=True)
        if bad=='master':state['parameters']['w'].add_(1e-7)
        else:state['optimizer']['state'][0]['exp_avg'].add_(1e-7)
        torch.save(state,p)
    elif bad=='before':atomic_json(out/'before.json',{'same':False})
    elif bad=='backend':
        p=s/'run/vllm/result.json';data=module.read(p);data['records'][0]['candidate_logprobs']=[[-.7,-.6]];atomic_json(p,data)
    elif bad in ('phase','rank'):
        p=out/'result.json';data=module.read(p);data[bad]='accepted' if bad=='phase' else 2;atomic_json(p,data)
    else:atomic_json(r/'run/resident-selection.json',{'same':False})
    with pytest.raises(ValueError):module.verify(r,s)
