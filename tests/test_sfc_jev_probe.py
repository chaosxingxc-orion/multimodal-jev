import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_sfc_jev import load_tool
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def setup(tmp_path,monkeypatch):
    module=load_tool('infer_sfc_jev_omni_probe')
    rows=[dict(id=str(i),input={'decisions':[{'id':str(j)} for j in range(13)]}) for i in range(6)]
    path=tmp_path/'train_id.requests.jsonl';path.write_bytes(b''.join(map(canonical,rows)))
    (tmp_path/'MANIFEST.json').write_text('{}')
    monkeypatch.setattr(module,'select',lambda actual:actual)
    run=tmp_path/'preflight/run';run.mkdir(parents=True)
    identity=dict(model='/model',source_files={},release=str(tmp_path),manifest_sha256=file_hash(tmp_path/'MANIFEST.json'),
        train_requests_sha256=file_hash(path),selected_ids=[r['id'] for r in rows])
    (run/'identity.json').write_bytes(canonical(identity))
    result=dict(status='passed',decisions=78,bundles=6,identity_sha256=file_hash(run/'identity.json'),
        records=[dict(id=r['id'],decision_id=d['id'],input_tokens=100,expected_frames=[30],observed_frames=[30])
                 for r in rows for d in r['input']['decisions']])
    (run/'result.json').write_bytes(canonical(result));(run/'terminal.json').write_text('{"exit_code":0}')
    cfg=dict(split='train_id',parameters_frozen=True,route='audio-direct',dtype='bfloat16',seed=17,
        max_context_tokens=4096,max_new_tokens=1024,sampling={'temperature':0.,'n':1},audio_model='/model',
        processor_preflight=str(run.parent),processor_preflight_sha256=file_hash(run/'result.json'))
    return module,cfg,run,result


def test_complete_preflight_requests_only(tmp_path,monkeypatch):
    module,cfg,run,result=setup(tmp_path,monkeypatch)
    rows,identity,checks=module.select_inputs(cfg)
    assert len(rows)==6 and len(checks)==78
    assert not (tmp_path/'train_id.targets.jsonl').exists()


@pytest.mark.parametrize('change',['test','weights','budget','prefix','missing','order','frames','request','source'])
def test_preflight_drift_rejected(tmp_path,monkeypatch,change):
    module,cfg,run,result=setup(tmp_path,monkeypatch)
    if change=='test':cfg['split']='test_id'
    if change=='weights':cfg['parameters_frozen']=False
    if change=='budget':cfg['max_new_tokens']=128
    if change=='prefix':cfg['adapter']='trained'
    if change=='missing':result['records'].pop()
    if change=='order':result['records'].reverse()
    if change=='frames':result['records'][0]['observed_frames']=[29]
    if change=='request':(tmp_path/'train_id.requests.jsonl').write_text('changed')
    if change=='source':(tmp_path/'MANIFEST.json').write_text('changed')
    (run/'result.json').write_bytes(canonical(result));cfg['processor_preflight_sha256']=file_hash(run/'result.json')
    with pytest.raises(ValueError):module.select_inputs(cfg)


def response():
    return SimpleNamespace(finished=True,prompt_token_ids=[1,2],outputs=[SimpleNamespace(text='not valid JSON',token_ids=[3],finish_reason='stop')])


def test_engine_gate_accepts_quality_failure_but_not_input_or_termination_drift():
    module=load_tool('infer_sfc_jev_omni_probe')
    assert module.accepted_response(response(),[1,2]).text=='not valid JSON'
    for change in ('ids','unfinished','multiple','budget','reason'):
        r=response()
        if change=='ids':r.prompt_token_ids=[2,1]
        if change=='unfinished':r.finished=False
        if change=='multiple':r.outputs*=2
        if change=='budget':r.outputs[0].token_ids=[3]*1025
        if change=='reason':r.outputs[0].finish_reason='abort'
        with pytest.raises(ValueError):module.accepted_response(r,[1,2])


def test_independent_acceptance_rejects_changed_prediction_and_missing_decision(tmp_path,monkeypatch):
    module=load_tool('verify_sfc_jev_probe');run=tmp_path/'run';run.mkdir()
    rows=[dict(id=str(i),source_group=str(i%4),audio_sha256=['audio'],input={'decisions':[{'id':str(j)} for j in range(13)]}) for i in range(6)]
    checks={(r['id'],d['id']):dict(input_tokens=100,input_ids_sha256='ids',prompt_sha256='prompt',expected_frames=[30])
            for r in rows for d in r['input']['decisions']}
    monkeypatch.setattr(module,'select_inputs',lambda cfg:(rows,{},checks))
    monkeypatch.setattr(module,'SOURCE_FILES',('src/check.py',))
    code=tmp_path/'source/src/check.py';code.parent.mkdir(parents=True);code.write_text('pinned')
    write=lambda name,value:(tmp_path/name).write_bytes(canonical(value))
    write('audit.json',{'files':['model']})
    cfg=dict(asset_audit=str(tmp_path/'audit.json'),versions={'vllm':'pinned'},processor_preflight_sha256='preflight')
    write('config.json',cfg)
    write('preparation.json',dict(image='image',source_commit='commit',config_sha256=file_hash(tmp_path/'config.json'),inference_files={'src/check.py':file_hash(code)}))
    write('launch.json',dict(container_id='cid',image='image',source_commit='commit',physical_devices=[6,7]))
    monkeypatch.setattr(module,'inspect',lambda cid:dict(Id='cid',Image='image',State={'Status':'exited','ExitCode':0}))
    write('run/identity.json',dict(config=cfg,model_files=['model'],versions=cfg['versions'],inputs_sha256=module.digest(rows),
        processor_preflight_sha256='preflight',source_files={'src/check.py':file_hash(code)}))
    write('run/terminal.json',{'exit_code':0});write('run/selection.json',dict(ids=[r['id'] for r in rows],decisions=[list(k) for k in checks]))
    predictions=[dict(example_id=r['id'],decision_id=d['id'],request_sha256=module.digest(r),source_group=r['source_group'],
        audio_sha256=r['audio_sha256'],split='train_id',hf_vllm_prompt_ids_equal=True,raw_text='invalid output',
        token_ids=[1],finish_reason='stop',length_limited=False,**checks[r['id'],d['id']])
        for r in rows for d in r['input']['decisions']]
    def save():
        path=run/'predictions.jsonl';path.write_bytes(b''.join(map(canonical,predictions)))
        write('run/result.json',dict(status='passed',weight_updates=0,model_files_unchanged=True,
            predictions_sha256=file_hash(path),decisions=78,predictions=78,bundles=6,length_limited=0))
    save();assert len(module.accept(tmp_path)[2])==78
    predictions[0]['input_ids_sha256']='changed';save()
    with pytest.raises(ValueError,match='parity'):module.accept(tmp_path)
    predictions.pop();save()
    with pytest.raises(ValueError,match='78 decisions'):module.accept(tmp_path)
