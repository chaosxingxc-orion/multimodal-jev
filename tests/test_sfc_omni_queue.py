import json
import hashlib
from pathlib import Path
import sys
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import watch_sfc_omni_probe as module


def test_both_predecessors_then_validation_before_single_dispatch(tmp_path,monkeypatch):
    plan=dict(dependencies={'native_test':{'root':'native'},'transfer_train':{'root':'transfer'}})
    (tmp_path/'queue-plan.json').write_bytes(canonical(plan));(tmp_path/'config.json').write_text('{}')
    events=[];checks=iter([True,False,True,True])
    monkeypatch.setattr(module,'check_pins',lambda *args:events.append('pins'))
    monkeypatch.setattr(module,'observe_dependency',lambda spec:next(checks))
    monkeypatch.setattr(module.time,'sleep',lambda seconds:events.append('wait'))
    monkeypatch.setattr(module,'verify_transfer',lambda root:events.append('verify'))
    monkeypatch.setattr(module,'select_inputs',lambda cfg:events.append('inputs'))
    monkeypatch.setattr(module,'dispatch_once',lambda *args:events.append('dispatch'))
    monkeypatch.setattr(module,'accept_and_score',lambda *args:events.append('score'))
    module.execute(tmp_path)
    assert events==['pins','wait','pins','verify','inputs','dispatch','score']


def test_failed_predecessor_does_not_dispatch(tmp_path,monkeypatch):
    (tmp_path/'queue-plan.json').write_bytes(canonical(dict(dependencies={'native_test':{},'transfer_train':{}})))
    monkeypatch.setattr(module,'check_pins',lambda *args:None)
    def failed(spec):raise RuntimeError('dependency failed')
    monkeypatch.setattr(module,'observe_dependency',failed)
    monkeypatch.setattr(module,'dispatch_once',lambda *args:pytest.fail('must not dispatch'))
    with pytest.raises(RuntimeError,match='dependency failed'):module.execute(tmp_path)


@pytest.mark.parametrize('prior',['dispatch-intent.json','dispatch.log','launch.json','run'])
def test_ambiguous_or_prior_dispatch_never_replayed(tmp_path,prior):
    (tmp_path/prior).touch()
    with pytest.raises(FileExistsError,match='no replay'):module.dispatch_once(tmp_path,{})


def test_busy_npu_prevents_dispatch(tmp_path,monkeypatch):
    (tmp_path/'preparation.json').write_bytes(canonical(dict(physical_devices=[6,7],image='image',
        argv=['docker','run','--device','/dev/davinci6:/dev/davinci6','--device','/dev/davinci7:/dev/davinci7'])))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:'occupied')
    monkeypatch.setattr(module,'inspect',lambda cid:{'HostConfig':{'Devices':[{'PathOnHost':'/dev/davinci6'}]}})
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('must not dispatch'))
    with pytest.raises(RuntimeError,match='occupied'):module.dispatch_once(tmp_path,{'image':'image'})
    assert not (tmp_path/'dispatch-intent.json').exists()


def test_config_pin_change_prevents_wait_or_dispatch(tmp_path):
    config=tmp_path/'config.json';config.write_text('original')
    plan={'files':{'config.json':file_hash(config)}};config.write_text('changed')
    with pytest.raises(ValueError,match='prepared SFC input changed'):module.check_pins(tmp_path,plan)


def test_complete_raw_outputs_are_scored_and_missing_turn_is_rejected(tmp_path,monkeypatch):
    root=tmp_path/'root';run=root/'run';run.mkdir(parents=True)
    write=lambda path,value:path.write_bytes(canonical(value))
    rows=[dict(example_id=str(i),input={'audios':['a']*(1 if i<4 else 2)},
        metadata={'source_group':str(i%4),'audio_sha256':['hash']*(1 if i<4 else 2)}) for i in range(6)]
    release=tmp_path/'release';release.mkdir()
    (release/'train_id.jsonl').write_bytes(b''.join(canonical(dict(r,supervision={'official_solution':'[]'})) for r in rows))
    monkeypatch.setattr(module,'select_inputs',lambda cfg:(rows,{'release':str(release)}))
    audit=root/'audit.json';write(audit,{'files':['fixed']})
    cfg=dict(asset_audit=str(audit),versions={'vllm':'fixed'});write(root/'config.json',cfg)
    write(run/'identity.json',dict(config=cfg,versions=cfg['versions'],model_files=['fixed'],source_files={},
        inputs_sha256=hashlib.sha256(canonical(rows)).hexdigest()))
    write(root/'launch.json',dict(container_id='cid'))
    monkeypatch.setattr(module,'inspect',lambda cid:dict(Id='cid',Image='image',State={'Status':'exited','ExitCode':0}))
    write(run/'terminal.json',{'exit_code':0});write(run/'selection.json',{'ids':[r['example_id'] for r in rows]})
    preds=[dict(example_id=r['example_id'],split='train_id',source_group=r['metadata']['source_group'],
        audio_count=len(r['input']['audios']),audio_sha256=r['metadata']['audio_sha256'],
        input_sha256=hashlib.sha256(canonical(r['input'])).hexdigest(),hf_vllm_prompt_ids_equal=True,
        input_tokens=1000,token_ids=[1],raw_text='invalid JSON',finish_reason='stop',length_limited=False) for r in rows]
    result=dict(status='passed',weight_updates=0,model_files_unchanged=True,requests=6,predictions=6,length_limited=0)
    def save():
        (run/'predictions.jsonl').write_bytes(b''.join(map(canonical,preds)))
        result['predictions_sha256']=file_hash(run/'predictions.jsonl');write(run/'result.json',result)
    save();plan=dict(image='image',official_scorer='/scorer',official_scorer_sha256='fixed')
    def scorer(manifest,predictions,official,output,digest):
        assert len(manifest.read_text().splitlines())==6
        assert all(json.loads(l)['raw_text']=='invalid JSON' for l in predictions.read_text().splitlines())
        output.mkdir();report=dict(strict_metrics={'all':{'exact_correct':0}},strict_conversation_metrics={})
        write(output/'metrics.json',report);return report
    monkeypatch.setattr(module,'evaluate',scorer)
    module.accept_and_score(root,plan)
    assert json.loads((root/'acceptance.json').read_text())['requests']==6
    preds.pop();save()
    with pytest.raises(ValueError,match='all six'):module.accept_and_score(root,plan)
