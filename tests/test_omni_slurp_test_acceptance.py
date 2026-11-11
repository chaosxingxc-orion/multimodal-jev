import copy
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import pytest

from multimodal_jev.consumption import canonical
from multimodal_jev.baseline import file_hash

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import verify_omni_slurp_test as module


def runtime_fixture():
    launch=dict(image='image',argv=['docker','run','-v','/source:/source:ro',
        '-e','HF_HUB_OFFLINE=1','image','timeout','86400s','python3','/source/run.py'])
    state=dict(Status='exited',ExitCode=0,OOMKilled=False,Dead=False,Error='')
    paths=['/dev/davinci6','/dev/davinci7','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']
    item=dict(Id='cid',Image='image',State=state,RestartCount=0,
        Config=dict(Cmd=launch['argv'][launch['argv'].index('image')+1:],Env=['HF_HUB_OFFLINE=1']),
        HostConfig=dict(Privileged=False,NetworkMode='host',NanoCpus=16*10**9,
            Memory=128*1024**3,ShmSize=8*1024**3,RestartPolicy={'Name':'no'},
            Binds=['/source:/source:ro'],Devices=[dict(PathOnHost=p,PathInContainer=p,
                CgroupPermissions='rwm') for p in paths]))
    return item,launch


@pytest.mark.parametrize('change',['oom','dead','error','privileged','network','cpu','memory',
    'shm','restart_policy','restarted','device','device_target','extra_device','device_request',
    'device_rule','mount','extra_mount','command','env','duplicate_env'])
def test_runtime_rejects_execution_drift(change):
    item,launch=runtime_fixture()
    module.validate_runtime(item,launch)
    host=item['HostConfig']
    if change=='oom':item['State']['OOMKilled']=True
    if change=='dead':item['State']['Dead']=True
    if change=='error':item['State']['Error']='runtime failure'
    if change=='privileged':host['Privileged']=True
    if change=='network':host['NetworkMode']='none'
    if change=='cpu':host['NanoCpus']=8*10**9
    if change=='memory':host['Memory']=0
    if change=='shm':host['ShmSize']=64*1024**2
    if change=='restart_policy':host['RestartPolicy']['Name']='always'
    if change=='restarted':item['RestartCount']=1
    if change=='device':host['Devices'][0]['PathOnHost']='/dev/davinci4'
    if change=='device_target':host['Devices'][0]['PathInContainer']='/dev/davinci4'
    if change=='extra_device':host['Devices'].append(dict(host['Devices'][0]))
    if change=='device_request':host['DeviceRequests']=[{'Count':-1}]
    if change=='device_rule':host['DeviceCgroupRules']=['c *:* rwm']
    if change=='mount':host['Binds']=['/source:/source:rw']
    if change=='extra_mount':host['Binds'].append('/other:/other:ro')
    if change=='command':item['Config']['Cmd']=['true']
    if change=='env':item['Config']['Env']=['HF_HUB_OFFLINE=0']
    if change=='duplicate_env':item['Config']['Env'].append('HF_HUB_OFFLINE=0')
    with pytest.raises(ValueError):module.validate_runtime(item,launch)


def fixture():
    rows=[dict(id=str(i),split='test',source_group=str(i%2974)) for i in range(13078)]
    predictions=[dict(example_id=r['id'],source_group=r['source_group'],split='test',replicate=0,
        request_sha256=hashlib.sha256(canonical(r)).hexdigest(),audio_condition='actual',
        input_tokens=1000,token_ids=[1],finish_reason='stop',length_limited=False,seconds=1.) for r in rows]
    return rows,predictions,dict(requests=13078,predictions=13078,length_limited=0),dict(max_new_tokens=256,max_context_tokens=4096)


def test_complete_population_and_length_limited_results_are_accepted():
    args=fixture()
    module.validate_records(*args)
    args[1][0].update(token_ids=[1]*256,finish_reason='length',length_limited=True)
    args[2]['length_limited']=1
    module.validate_records(*args)


@pytest.mark.parametrize('change',['missing','duplicate','binding','group','split','bool_replicate','context','output','duration','truncation'])
def test_native_test_accounting_rejects_incomplete_or_misattributed_outputs(change):
    args=fixture();preds=args[1]
    if change=='missing':preds.pop()
    if change=='duplicate':preds[-1]=copy.deepcopy(preds[0])
    if change=='binding':preds[0]['request_sha256']='wrong'
    if change=='group':preds[0]['source_group']='wrong'
    if change=='split':preds[0]['split']='train'
    if change=='bool_replicate':preds[0]['replicate']=False
    if change=='context':preds[0]['input_tokens']=4000
    if change=='output':preds[0]['token_ids']=[1]*257
    if change=='duration':preds[0]['seconds']=float('nan')
    if change=='truncation':args[2]['length_limited']=1
    with pytest.raises(ValueError):module.validate_records(*args)


def test_scorer_rounding_is_disclosed_but_wrong_counts_and_metrics_fail():
    original={'metrics':{'f1':[.3331405436668594,109.05555555555557]},'examples':13078}
    rescored={'metrics':{'f1':[.33314054366685947,109.05555555555556]},'examples':13078}
    differences=module.compare_values(original,rescored)
    assert len(differences)==2
    assert differences[0]['path']=='/metrics/f1/0'
    for changed in [dict(rescored,examples=13077),dict(rescored,examples=13078.),
                    dict(rescored,metrics={'f1':[.3332,109.05555555555556]}),
                    dict(rescored,metrics={'f1':[float('nan'),109.05555555555556]})]:
        with pytest.raises(ValueError):module.compare_values(original,changed)


def test_running_or_failed_job_is_not_rescored(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'score_run',lambda *args:pytest.fail('must not read incomplete TEST'))
    args=(tmp_path,tmp_path/'launch-record.json',tmp_path/'probe',tmp_path/'acceptance')
    with pytest.raises(FileNotFoundError):module.verify(*args)
    (tmp_path/'evaluation-terminal.json').write_text(json.dumps({'exit_code':1}))
    with pytest.raises(ValueError,match='successful supervisor'):module.verify(*args)
    assert not (tmp_path/'acceptance').exists()


def test_acceptance_orchestration_with_mocked_docker_and_scorer(tmp_path,monkeypatch):
    # Real identity/source/hash/coverage gates; synthetic predictions and scorer.
    rows,preds,result,cfg=fixture();root=tmp_path/'test';run=root/'run';run.mkdir(parents=True)
    probe=tmp_path/'probe';(probe/'run').mkdir(parents=True)
    write=lambda p,d:p.write_bytes(canonical(d))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source=root/'source'
    names=['tools/jev/infer_qwen3_omni_probe.py','src/multimodal_jev/omni_slurp_test_contract.py']
    digests={}
    for name in names:
        dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(subprocess.check_output(['git','show',commit+':'+name]));digests[name]=file_hash(dest)
    cfg.update(audio_model='/model',dtype='bfloat16',seed=17,sampling={'temperature':0,'n':1},
        route='audio-direct',task_format='slurp-native',model_family='qwen3_omni_moe')
    for key in ['instruction','asset_audit']:
        path=root/key;path.write_text('fixed');cfg[key]=str(path)
    requests=root/'test.requests.jsonl';requests.write_bytes(b''.join(map(canonical,rows)));cfg['requests']=str(requests)
    write(root/'config.json',cfg)
    identity=dict(config=cfg,source_sha256={},runner_sha256=digests[names[0]],
        population_selector_sha256=digests[names[1]],instruction_sha256=file_hash(cfg['instruction']),
        asset_audit_sha256=file_hash(cfg['asset_audit']),requests_sha256=file_hash(requests),
        versions={'vllm':'fixed'},model_files=['fixed'],taxonomy_sha256='fixed')
    write(run/'identity.json',identity);write(probe/'run/identity.json',identity)
    for path in [root/'evaluation-terminal.json',run/'terminal.json',probe/'evaluation-terminal.json']:write(path,{'exit_code':0})
    (run/'predictions.jsonl').write_bytes(b''.join(map(canonical,preds)))
    result.update(status='passed',weight_updates=0,model_files_unchanged=True,audio_condition='actual',
        predictions_sha256=file_hash(run/'predictions.jsonl'),format_valid=0,schema_valid=0)
    write(run/'result.json',result);write(run/'selection.json',{'ids':[r['id'] for r in rows]})
    launch=dict(container_id='cid',image='image',physical_devices=[6,7],source_commit=commit,
        config_sha256=file_hash(root/'config.json'),scorer_repository='/scorer')
    item,runtime_launch=runtime_fixture();launch['argv']=runtime_launch['argv']
    write(root/'launch.json',launch)
    write(root/'docker-terminal.json',dict(container_id='cid',state=item['State']))
    record=tmp_path/'launch-record.json'
    write(record,{'jobs':[dict(launch,root=str(root),supervisor={'source_commit':commit})]})
    monkeypatch.setattr(module,'inspect_container',lambda cid:item)
    monkeypatch.setattr(module,'select_native_test',lambda c,r:r)
    report=dict(examples=13078,full_official_split=True,format_valid=0,schema_valid=0,
                metrics={'f1':0.,'action':{'invalid':(0.,0.,0.,0,3,0)}})
    (root/'score').mkdir();write(root/'score/result.json',report)
    (root/'score/official-predictions.jsonl').write_text('synthetic normalization\n')
    def score(*args):
        out=args[-1];out.mkdir();write(out/'result.json',report)
        (out/'official-predictions.jsonl').write_text('synthetic normalization\n');return report
    monkeypatch.setattr(module,'score_run',score)
    output=tmp_path/'acceptance'
    accepted=module.verify(root,record,probe,output)
    assert accepted['examples']==13078 and accepted['source_groups']==2974
    assert accepted['normalized_predictions_exact'] and accepted['score_float_differences']==[]
    with pytest.raises(FileExistsError):module.verify(root,record,probe,output)
    identity['versions']['vllm']='changed';write(run/'identity.json',identity)
    with pytest.raises(ValueError,match='TRAIN baseline identity'):module.verify(root,record,probe,tmp_path/'drift')
    assert not (tmp_path/'drift').exists()
