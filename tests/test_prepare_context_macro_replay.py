"""Recovery packaging tests use synthetic artifacts, not training evidence."""
import copy
import json
from pathlib import Path

import pytest

from tools.reports import prepare_context_macro_replay as m
from tools.reports.context_macro_training_operations import validate, command
from tools.reports.context_macro_trainer_candidate import ALGORITHM
from tools.reports.prepare_context_batch_probe_candidate import IMAGE


def fixture(tmp_path):
    root='/local/prefix';prepared=tmp_path/'prefix';p=prepared/'package';p.mkdir(parents=True)
    ops=tmp_path/'operations';run=ops/'collected/run';run.mkdir(parents=True)
    client=dict(max_queries=128,rollout_queries=32,journal_root=root+'/journal',attempt_id='prefix',
        runtime_identity={'profile':'fixture'},replicas=[dict(socket=root+f'/ipc/r{i}.sock',exchange=root+f'/ipc/e{i}') for i in range(2)])
    t={k:k for k in ('base_id','source_policy_id','source_contract','runtime_profile','release_sha256','schedule_sha256','behavior_bounds')}
    t.update(stage='task',algorithm=ALGORITHM,dual_arm='joint',physical_devices=[4,5],global_batch=128,microbatch=8,
        start_update=0,stop_update=2,checkpoint_interval=1,engineering_prefix=True,training_dispatch_allowed=False,
        source_root=root+'/package',source_policy=root+'/package/policy',initial_adapter=root+'/package/adapter',
        reference_adapter=root+'/package/adapter',reference_cache=root+'/reference-cache',model='/shared/model',
        encoded_audio_cache='/audio/cache',audio_mirror_root='/mirror',macro_client=client,
        dual_runtime_identity=client['runtime_identity'],pins={'/shared/data':'external-hash'})
    (p/'model_code.py').write_text('# immutable runtime\n')
    m.write(p/'RUNTIME-SOURCES.json',{'model_code.py':m.file_hash(p/'model_code.py')})
    for name in ('model_code.py','RUNTIME-SOURCES.json'):t['pins'][root+'/package/'+name]=m.file_hash(p/name)
    m.write(p/'training.json',t)
    for i in range(2):
        service=dict(policy=dict(physical_devices=[6+i],snapshot=t['source_policy'],adapter=t['initial_adapter'],
            policy_id=t['source_policy_id'],contract=t['source_contract']),pins={},
            dual_runtime_identity=t['dual_runtime_identity'],**client['replicas'][i])
        service['socket_path']=service.pop('socket');m.write(p/f'service-{i}.json',service)
    m.write(p/'FILES.json',{str(x.relative_to(p)):m.file_hash(x) for x in p.iterdir()})
    parent=dict(prepared=str(prepared),operations_root=str(ops),remote_root=root,
        model='/shared/model',host='npu',tag='prefix',shared_root='/shared',driver_root='/driver',manifest_sha256=m.verify(p))
    m.write(prepared/'PREPARED.json',dict(status='macro_training_prefix_prepared_only',image=IMAGE,files_sha256=m.verify(p),
        remote_root=root,ipc_root=root+'/ipc',runtime_identity=t['dual_runtime_identity'],training_dispatch_allowed=False))
    for name in ('rank-0/checkpoint-0001.pt','rank-1/checkpoint-0001.pt','policy-0001/execution.pt','adapter-0001/adapter.safetensors'):
        x=run/name;x.parent.mkdir(parents=True,exist_ok=True);x.write_bytes(b'synthetic recovery asset')
    commit=dict(updates=1,contract=m.expected_contract(t),rank_shards={f'rank-{i}/checkpoint-0001.pt':m.file_hash(run/f'rank-{i}/checkpoint-0001.pt') for i in range(2)},joint_policy={'policy_id':'step1'})
    m.write(run/'checkpoint-0001-committed.json',commit)
    m.write(ops/'transport.json',dict(files={str(x.relative_to(ops/'collected')):m.file_hash(x) for x in run.rglob('*') if x.is_file()}))
    admission=dict(status='finite_dp2_training_audited',files_sha256=m.verify(p),fresh_process_replay_verified=False,containers={'client':'a'*64})
    c=dict(output=str(tmp_path/'replay'),remote_root='/local/replay')
    return c,parent,admission


def test_preserves_runtime_and_contract_only_copies_delta_recovery(tmp_path):
    c,parent,a=fixture(tmp_path);r=m.seal(c,{},parent,a);p=Path(c['output'])/'package'
    t=json.loads((p/'training.json').read_text());old=json.loads((Path(parent['prepared'])/'package/training.json').read_text())
    assert m.expected_contract(t)==m.expected_contract(old)
    assert t['start_update']==1 and t['resume_root']=='/local/replay/package/resume'
    assert (p/'RUNTIME-SOURCES.json').read_bytes()==(Path(parent['prepared'])/'package/RUNTIME-SOURCES.json').read_bytes()
    assert len(list((p/'resume').rglob('checkpoint-0001.pt')))==2
    for i in range(2):
        s=json.loads((p/f'service-{i}.json').read_text())
        assert s['policy']['policy_id']=='step1' and '/resume/' in s['policy']['snapshot']
        assert '/shared/data' not in s['pins']
    ops=dict(parent,prepared=c['output'],remote_root=c['remote_root'],manifest_sha256=r['files_sha256'])
    assert validate(ops)['status']=='macro_training_replay_prepared_only'
    code=command(ops,'preflight')[-1]
    assert "verify_native_adapter(c['reference_adapter'],b)" in code
    assert "verify_native_adapter(c['initial_adapter'],rb)" in code
    compile(code,'preflight','exec')


@pytest.mark.parametrize('fault',['parent','asset','runtime','contract','root'])
def test_unauthenticated_or_changed_recovery_rejected(tmp_path,fault):
    c,parent,a=fixture(tmp_path)
    if fault=='parent':a['status']='unverified'
    if fault=='asset':(Path(parent['operations_root'])/'collected/run/rank-1/checkpoint-0001.pt').write_bytes(b'changed')
    if fault=='runtime':(Path(parent['prepared'])/'package/model_code.py').write_text('changed')
    if fault=='contract':
        p=Path(parent['operations_root'])/'collected/run/checkpoint-0001-committed.json'
        value=json.loads(p.read_text());value['contract']['world_size']=4;m.write(p,value)
    if fault=='root':c['remote_root']='/local/prefix/child'
    with pytest.raises(ValueError):m.seal(c,{},parent,a)


def test_public_prepare_reaudits_instead_of_trusting_a_pass_flag(tmp_path,monkeypatch):
    c,parent,a=fixture(tmp_path)
    monkeypatch.setattr(m,'load_config',lambda *args:(parent,{}))
    def reject(_):raise ValueError('actual parent audit fails')
    monkeypatch.setattr(m,'audit_collected',reject)
    c.update(parent_config='template',parent_local='local')
    with pytest.raises(ValueError,match='actual parent'):m.prepare(c,{})
    assert not Path(c['output']).exists()
