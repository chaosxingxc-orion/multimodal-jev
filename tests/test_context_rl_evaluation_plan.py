import json
from pathlib import Path
import pytest
from tools.jev.context_rl_evaluation_plan import build,sha


def fixture(tmp):
    experiment=json.loads(Path('configs/releases/context-prob-v1/experiment.json').read_text())
    roots=[tmp/'stage1',tmp/'rl'];proofs=[];contract={'runtime':'fixture',
        'token':{'registration_status':'admitted'},'request_version':'jev-context-exclusive_event-v1'}
    for i,root in enumerate(roots):
        root.mkdir();files={}
        def put(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));files[name]=sha(p);return p
        steps=[634] if i==0 else experiment['evaluation']['checkpoints'][1:]
        for step in steps:
            directory=f'policy-{step:04d}';pid='initial' if i==0 else f'rl-{step}'
            asset=put('run/'+directory+'/execution.pt',{'fixture':pid})
            manifest=put('run/'+directory+'/MANIFEST.json',dict(policy_id=pid,contract=contract,files={'execution.pt':sha(asset)}))
            put(f'run/checkpoint-{step:04d}-committed.json',dict(updates=step,joint_policy_directory=directory,
                joint_policy={'policy_id':pid},joint_policy_manifest_sha256=sha(manifest),
                contract=dict(stage='instruction' if i==0 else 'task',algorithm='grpo_v1',source_policy_id='initial',release_sha256=experiment['release_sha256'])))
        if i:
            put('config.json',dict(stage='task',algorithm='grpo_v1',engineering_prefix=False,
                training_dispatch_allowed=True,start_update=0,stop_update=634,
                release_sha256=experiment['release_sha256'],source_policy_id='initial',source_contract=contract))
        else:
            put('config.json',dict(stage='instruction',engineering_prefix=False,stop_update=634,
                                  release_sha256=experiment['release_sha256']))
        proofs.append(dict(passed=True,state=dict(Status='exited',ExitCode=0,OOMKilled=False),files=files))
    return dict(experiment=experiment,stage1_root=roots[0],stage1_proof=proofs[0],
                rl_root=roots[1],rl_proof=proofs[1],run_id='test',algorithm='grpo_v1')


def test_complete_source_binding(tmp_path):
    plan=build(**fixture(tmp_path))
    assert len(plan['checkpoints'])==11
    assert plan['checkpoints'][0]['source_stage']=='instruction'
    assert all(x['source_stage']=='task' for x in plan['checkpoints'][1:])
    assert plan['initial_policy_id']=='initial'


@pytest.mark.parametrize('fault',['missing','mutated','wrong_algorithm','wrong_reference','prefix','live','wrong_step','stage1_is_rl'])
def test_reject_invalid_curve_source(tmp_path,fault):
    args=fixture(tmp_path);root=args['rl_root'];proof=args['rl_proof']
    if fault=='missing':del proof['files']['run/checkpoint-0064-committed.json']
    elif fault=='mutated':(root/'run/policy-0064/execution.pt').write_text('changed')
    elif fault=='live':proof['state']['Status']='running'
    elif fault=='stage1_is_rl':
        p=args['stage1_root']/'config.json';c=json.loads(p.read_text());c['stage']='task'
        p.write_text(json.dumps(c));args['stage1_proof']['files']['config.json']=sha(p)
    else:
        name='config.json' if fault in ('wrong_algorithm','wrong_reference','prefix') else 'run/checkpoint-0064-committed.json'
        p=root/name;c=json.loads(p.read_text())
        if fault=='wrong_algorithm':c['algorithm']='rloo_v1'
        if fault=='wrong_reference':c['source_policy_id']='other'
        if fault=='prefix':c['engineering_prefix']=True
        if fault=='wrong_step':c['updates']=128
        p.write_text(json.dumps(c));proof['files'][name]=sha(p)
    with pytest.raises(ValueError):build(**args)
