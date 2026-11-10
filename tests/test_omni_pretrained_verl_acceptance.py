from copy import deepcopy
from pathlib import Path
import sys
import json

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from verify_omni_pretrained_verl import check_probabilities, accept, sha


def fixture():
    reference=[[-1.,-.01] for _ in range(6)]
    micros=[dict(log_probs=[[-1.+v,-.01] for _ in range(6)],seconds=2.,max_error_from_unpadded=v)
            for v in (.02,.03)]
    comparison=dict(atol=.1,passed=True,errors=dict(microbatch=.01,unpadded=[.02,.03]))
    return reference,micros,comparison,deepcopy(comparison)


def test_recompute_from_raw_values():
    ref,micro,comparison,result=fixture()
    errors=check_probabilities(ref,micro,comparison,result)
    assert errors['unpadded']==pytest.approx([.02,.03])


@pytest.mark.parametrize('kind',['missing','extra','short','nan','positive','bool','reported',
                                'relaxed','failed','threshold','seconds','between','reference'])
def test_reject_partial_or_misreported_comparison(kind):
    ref,micro,comparison,result=fixture()
    if kind=='missing':micro[0]['log_probs'].pop()
    if kind=='extra':micro[0]['log_probs'].append([-1.,-.01])
    if kind=='short':micro[0]['log_probs'][0].pop()
    if kind in ('nan','positive','bool'):micro[0]['log_probs'][0][0]={'nan':float('nan'),'positive':.1,'bool':False}[kind]
    if kind=='reported':micro[0]['max_error_from_unpadded']=0
    if kind=='relaxed':result['atol']=.2
    if kind=='failed':comparison['passed']=False
    if kind=='threshold':micro[0]['log_probs'][0][0]=-.8
    if kind=='seconds':micro[0]['seconds']=0
    if kind=='between':result['errors']['microbatch']=0
    if kind=='reference':ref[0][0]=-2.
    with pytest.raises(ValueError):check_probabilities(ref,micro,comparison,result)


def complete_fixture(tmp):
    root=tmp/'current';prior=tmp/'prior';run=root/'run';run.mkdir(parents=True)
    (prior/'run').mkdir(parents=True);(root/'source/tools/jev').mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    source=root/'source/tools/jev/probe.py';source.write_text('# synthetic fixture, never executed\n')
    sources={'tools/jev/probe.py':sha(source)};cfg={'fixture':True}
    keys=[[str(i),'present'] for i in range(6)];order=[2,0,5,1,3,4]
    ref,micros,comparison,result=fixture()
    write(prior/'config.json',cfg);write(prior/'launch.json',{'fixture':True});write(prior/'intent.json',{'fixture':True})
    write(prior/'run/identity.json',dict(selected_keys=keys,config=cfg,model_files=[{'fixture':True}]))
    write(prior/'run/result.json',dict(identity_sha256=sha(prior/'run/identity.json'),responses=[
        dict(audio_count=c,response_token_ids=[3849,151645],response_log_probs=ref[i])
        for i,c in enumerate([1,1,2,1,1,2])]))
    accepted=tmp/'accepted.json'
    write(accepted,dict(schema='omni-pretrained-cpu-forward-acceptance-v1',forwards=6,weight_updates=0,
        result_sha256=sha(prior/'run/result.json'),launch_sha256=sha(prior/'launch.json'),
        intent_sha256=sha(prior/'intent.json'),container_id='b'*64))
    write(root/'inspection.json',dict(Id='b'*64,Image='image',State=dict(Status='exited',ExitCode=0)))
    write(root/'preflight-evidence.json',dict(result=dict(status='passed',preflight_only=True,model_loaded=False,
        tensor_shape=[6,1230],order=order,weight_updates=0)))
    argv=['docker','create','image','timeout','14400s','python','probe.py','--prior',str(prior),
        '--acceptance',str(accepted),'--acceptance-hash',sha(accepted),'--inspection',str(root/'inspection.json'),
        '--inspection-hash',sha(root/'inspection.json'),'--output',str(run)]
    write(root/'intent.json',dict(argv=argv,image='image',automatic_retry=False,wall_seconds=14400,
        atol=.1,micro_batch_sizes=[1,2],weight_updates=0,source_commit='commit',source_files=sources,
        acceptance_sha256=sha(accepted),inspection_sha256=sha(root/'inspection.json'),
        preflight_evidence_sha256=sha(root/'preflight-evidence.json')))
    write(root/'launch.json',dict(container_id='a'*64,image='image',source_commit='commit',intent_sha256=sha(root/'intent.json')))
    state=dict(Id='a'*64,Image='image',State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        HostConfig=dict(Devices=[],Privileged=False,NetworkMode='none',Memory=160*1024**3,NanoCpus=16000000000),
        Config=dict(Cmd=argv[3:]))
    inspection=tmp/'inspection.json';write(inspection,state)
    write(run/'identity.json',dict(order=order,keys=[keys[i] for i in order],response_ids=[3849,151645],
        config=cfg,model_files=[{'fixture':True}],source_files=sources,prior_acceptance_sha256=sha(accepted),
        inspection_sha256=sha(root/'inspection.json'),targets_read=False,synthetic_old_log_probs=True))
    write(run/'loading-info.json',dict(missing_keys=[],mismatched_keys=[],error_msgs=[],unexpected_keys=['talker.test']))
    result.update(status='passed',preflight_only=False,model_loaded=True,weight_updates=0,elapsed_seconds=10.,peak_rss_kib=100)
    write(run/'result.json',result);write(run/'terminal.json',dict(exit_code=0));write(run/'comparison.json',comparison)
    for i,m in enumerate(micros,1):write(run/f'microbatch-{i}.json',m)
    return root,inspection


@pytest.mark.parametrize('change',[None,'running','failed','command','source','order','loading','preflight',
                                  'prior','targets','updates','time','identity','extra'])
def test_complete_artifact_chain(tmp_path,change):
    root,inspection=complete_fixture(tmp_path)
    def edit(path,fn):
        d=json.loads(path.read_text());fn(d);path.write_text(json.dumps(d))
    if change=='running':edit(inspection,lambda d:d['State'].update(Status='running'))
    if change=='failed':edit(inspection,lambda d:d['State'].update(ExitCode=1))
    if change=='command':edit(inspection,lambda d:d['Config']['Cmd'].append('--preflight-only'))
    if change=='source':(root/'source/tools/jev/probe.py').write_text('# changed')
    if change=='order':edit(root/'run/identity.json',lambda d:d['order'].reverse())
    if change=='loading':edit(root/'run/loading-info.json',lambda d:d['missing_keys'].append('model.layer'))
    if change=='preflight':edit(root/'preflight-evidence.json',lambda d:d['result'].update(model_loaded=True))
    if change=='prior':edit(tmp_path/'prior/run/result.json',lambda d:d['responses'].pop())
    if change=='targets':edit(root/'run/identity.json',lambda d:d.update(targets_read=True))
    if change=='updates':edit(root/'run/result.json',lambda d:d.update(weight_updates=1))
    if change=='time':edit(root/'run/result.json',lambda d:d.update(elapsed_seconds=1.))
    if change=='identity':edit(root/'run/identity.json',lambda d:d.update(model_files=[]))
    if change=='extra':(root/'run/microbatch-3.json').write_text('{}')
    if change is None:
        result=accept(root,sha(root/'launch.json'),inspection,sha(inspection))
        assert result['forwards_per_condition']==6 and result['weight_updates']==0
    else:
        with pytest.raises(ValueError):accept(root,sha(root/'launch.json'),inspection,sha(inspection))
