import json

import pytest

from tools import export_training_trajectory as module


def fixture(tmp_path):
    root=tmp_path/'collection';root.mkdir()
    files={}
    def put(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value));files[name]=module.sha(p)
    put('config.json',dict(physical_devices=[4,5,6,7],global_batch=2))
    for rank in range(4):
        put(f'run/rank-{rank}/update-0001.json',dict(update=1,rollout=dict(update=0,
            behavior_policy_id='b',fixed_reference_policy_id='r',queries=[
                dict(request_id=f'q{i}',behavior_record={'policy_id':'b'},reference_record={'policy_id':'r'}) for i in range(2)])))
    transport=dict(status='transport_complete_independent_audit_required',state={'Status':'exited'},
                   files=files,bad_pins=[],bad_package=[],launch_binding_mismatches=[])
    return root,transport


def test_replicated_global_queries_count_once(tmp_path):
    root,transport=fixture(tmp_path)
    result=list(module.updates(root,transport))
    assert len(result)==1 and len(result[0][1]['queries'])==2
    assert len(result[0][2])==4 and len(result[0][3])==4


@pytest.mark.parametrize('fault',['rank_missing','drift','replica','duplicate','policy','live'])
def test_invalid_trajectory_rejected(tmp_path,fault):
    root,transport=fixture(tmp_path)
    name='run/rank-3/update-0001.json';p=root/name
    if fault=='rank_missing':del transport['files'][name]
    elif fault=='live':transport['state']['Status']='running'
    else:
        value=json.loads(p.read_text())
        if fault in ('drift','replica'):value['rollout']['extra']='changed'
        elif fault=='duplicate':value['rollout']['queries'][1]['request_id']='q0'
        elif fault=='policy':value['rollout']['queries'][0]['behavior_record']['policy_id']='wrong'
        p.write_text(json.dumps(value))
        if fault!='drift':
            # Make all replicas equal for population/policy tests.
            targets=range(4) if fault in ('duplicate','policy') else [3]
            for rank in targets:
                n=f'run/rank-{rank}/update-0001.json';(root/n).write_text(json.dumps(value));transport['files'][n]=module.sha(root/n)
    with pytest.raises(ValueError):list(module.updates(root,transport))
