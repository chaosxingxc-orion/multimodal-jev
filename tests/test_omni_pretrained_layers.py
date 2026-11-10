from pathlib import Path
import sys
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from diagnose_omni_pretrained_layers import pair_inputs,complete_dependency
import json
import diagnose_omni_pretrained_layers as diagnostic
from multimodal_jev.baseline import file_hash


def test_pair_changes_only_masked_text_padding():
    item=dict(input_ids=torch.tensor([2,3,4]),input_features=torch.arange(12.).reshape(1,4,3),
              feature_attention_mask=torch.ones(1,3,dtype=torch.long))
    original=item['input_features'].clone()
    a,b=pair_inputs(item,[8,9],7,5,99)
    assert a['input_ids'].tolist()==[[2,3,4,8,9]]
    assert b['input_ids'].tolist()==[[99,99,99,99,2,3,4,8,9]]
    assert b['position_ids'][b['attention_mask'].bool()].tolist()==[0,1,2,3,4]
    assert torch.equal(a['input_features'],b['input_features'])
    assert torch.equal(item['input_features'],original)
    assert a['input_features'].dtype==torch.float32


def test_full_run_cannot_use_missing_terminal_evidence(tmp_path):
    with pytest.raises(ValueError,match='terminal inspection'):
        complete_dependency(tmp_path,None,None)


def dependency_fixture(root,monkeypatch):
    """Synthetic completed envelope; actual logit analysis is tested separately."""
    (root/'run').mkdir();source=root/'source';source.mkdir()
    runner=source/'runner.py';runner.write_text('# synthetic fixture\n')
    def write(path,data):path.write_text(json.dumps(data));return file_hash(path)
    identity_hash=write(root/'run/identity.json',dict(runner_sha256=file_hash(runner)))
    intent=dict(source_root=str(source),source_files={'runner.py':file_hash(runner)},
        source_commit='fixture',argv=['docker','create',diagnostic.IMAGE,'timeout','18000s','python3','-B',str(runner)])
    pin=write(root/'intent.json',intent)
    write(root/'launch.json',dict(container_id=diagnostic.PRECISION_CONTAINER,image=diagnostic.IMAGE,
        source_commit='fixture',intent_sha256=pin))
    write(root/'run/terminal.json',dict(exit_code=0))
    records=[dict(row=i//5,cell=i%5) for i in range(30)]
    files={}
    for i,record in enumerate(records):files[f'fixture-{i:02d}.json']=write(root/'run'/f'fixture-{i:02d}.json',record)
    write(root/'run/result.json',dict(status='diagnostic_complete',records=records,weight_updates=0))
    state=dict(Id=diagnostic.PRECISION_CONTAINER,Image=diagnostic.IMAGE,
        State=dict(Status='exited',ExitCode=0,OOMKilled=False),Config=dict(Cmd=intent['argv'][3:]),
        HostConfig=dict(NetworkMode='none',Devices=[],Privileged=False,Memory=160*1024**3,NanoCpus=16000000000))
    inspection=root/'inspection.json';write(inspection,state)
    monkeypatch.setattr(diagnostic,'IDENTITY_HASH',identity_hash)
    monkeypatch.setattr(diagnostic,'INTENT_HASH',pin)
    monkeypatch.setattr(diagnostic,'analyze',lambda *a:dict(completed_cells=30,record_sha256=files))
    return inspection,state


def test_complete_envelope_accepts_with_mocked_logit_audit(tmp_path,monkeypatch):
    inspection,state=dependency_fixture(tmp_path,monkeypatch)
    result=complete_dependency(tmp_path,inspection,file_hash(inspection))
    assert result['logits_audit']['completed_cells']==30


@pytest.mark.parametrize('fault',['running','failed','oom','command','resources','identity','source','records','weight_updates','intent'])
def test_rejects_dependency_mutations(tmp_path,monkeypatch,fault):
    inspection,state=dependency_fixture(tmp_path,monkeypatch)
    if fault=='running':state['State']['Status']='running'
    elif fault=='failed':state['State']['ExitCode']=1
    elif fault=='oom':state['State']['OOMKilled']=True
    elif fault=='command':state['Config']['Cmd']=['different']
    elif fault=='resources':state['HostConfig']['Privileged']=True
    elif fault=='identity':(tmp_path/'run/identity.json').write_text('{}')
    elif fault=='source':(tmp_path/'source/runner.py').write_text('changed')
    elif fault in ('records','weight_updates'):
        p=tmp_path/'run/result.json';r=json.loads(p.read_text())
        if fault=='records':r['records']=r['records'][:-1]
        else:r['weight_updates']=1
        p.write_text(json.dumps(r))
    elif fault=='intent':
        p=tmp_path/'intent.json';r=json.loads(p.read_text());r['source_commit']='changed';p.write_text(json.dumps(r))
        p=tmp_path/'launch.json';r=json.loads(p.read_text());r['intent_sha256']=file_hash(tmp_path/'intent.json');p.write_text(json.dumps(r))
    inspection.write_text(json.dumps(state))
    with pytest.raises(ValueError):complete_dependency(tmp_path,inspection,file_hash(inspection))
