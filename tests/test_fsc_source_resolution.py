from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/jev'))
import resolve_fsc_transfer_sources as module
from multimodal_jev.baseline import atomic_json, file_hash


def rule():
    points=[dict(key='step0',arm='shared_initial',update=0,source_gate='initial')]
    points += [dict(key=f'{a}-step{s}',arm=a,update=s,source_gate=f'{a}{s}') for a in ('D','Q','P') for s in (6,12)]
    return dict(schema='fsc-dqp-transfer-prospective-selection-v1',target_feedback_used_for_selection=False,checkpoints=points,pins={})


def test_exact_rule_allowed():
    module.validate_rule(rule())


@pytest.mark.parametrize('bad',['missing','order','step','duplicate','feedback'])
def test_no_posthoc_substitution(bad):
    value=rule()
    if bad=='missing':value['checkpoints'].pop()
    elif bad=='order':value['checkpoints'][1:3]=reversed(value['checkpoints'][1:3])
    elif bad=='step':value['checkpoints'][1]['update']=5
    elif bad=='duplicate':value['checkpoints'][1]['source_gate']='initial'
    else:value['target_feedback_used_for_selection']=True
    with pytest.raises(ValueError,match='seven-policy'):module.validate_rule(value)


@pytest.mark.parametrize('bad',['failed','incomplete','wrong_suite','wrong_arm'])
def test_controller_terminal_cannot_promote_incomplete_pilot(tmp_path,bad):
    (tmp_path/'run').mkdir()
    atomic_json(tmp_path/'run/terminal.json',dict(exit_code=1 if bad=='failed' else 0))
    atomic_json(tmp_path/'run/progress.json',dict(completed_updates=11 if bad=='incomplete' else 12,total_updates=12,
        accepted_suite='wrong' if bad=='wrong_suite' else 'suite',arm='Q' if bad=='wrong_arm' else 'P'))
    with pytest.raises(ValueError):module.controller_complete(tmp_path,'suite','P',{})


def test_no_source_or_target_processing_before_controllers_complete(tmp_path,monkeypatch):
    p=tmp_path/'rule.json';atomic_json(p,rule())
    cfg=dict(rule=str(p),pins={str(p):file_hash(p)})
    for key in ('d_controller','qp_controller'):
        root=tmp_path/key;root.mkdir();cfg[key]=str(root)
        for name in ('launch.json','config.json'):
            atomic_json(root/name,{})
            cfg['pins'][str(root/name)]=file_hash(root/name)
    monkeypatch.setattr(module,'controller_complete',lambda *a: (_ for _ in ()).throw(ValueError('still running')))
    monkeypatch.setattr(module,'verify_gate',lambda *a: pytest.fail('premature gate consumption'))
    with pytest.raises(ValueError,match='still running'):module.resolve(cfg)


def test_resolves_complete_fixed_matrix_and_rejects_changed_adapter(tmp_path,monkeypatch):
    value=rule();cfg=dict(pins={},image='image',base_id='base',runtime_profile='profile',pilot='pilot',manifest_sha256='manifest')
    for key,arm,last in (('d_controller','D','D12'),('qp_controller','P','P12')):
        root=tmp_path/key;root.mkdir();(root/'run').mkdir();cfg[key]=str(root)
        atomic_json(root/'config.json',dict(pins={}))
        atomic_json(root/'source-manifest.json',{})
        atomic_json(root/'launch.json',dict(config_sha256=file_hash(root/'config.json'),source_manifest_sha256=file_hash(root/'source-manifest.json')))
        atomic_json(root/'run/terminal.json',dict(exit_code=0))
        atomic_json(root/'run/progress.json',dict(completed_updates=12,total_updates=12,arm=arm,accepted_suite=str(tmp_path/last)))
        cfg['pins'].update({str(root/n):file_hash(root/n) for n in ('config.json','launch.json')})
    for item in value['checkpoints']:
        gate=tmp_path/item['source_gate'];gate.mkdir();(gate/'run').mkdir();item['source_gate']=str(gate)
        adapter=gate/'adapter';adapter.mkdir();(adapter/'adapter_model.safetensors').write_bytes(b'fixed fake weight bytes')
        adapter_id=module.checkpoint_identity(adapter)[0]
        c=dict(base_id='base',runtime_profile='profile',adapter=str(adapter),adapter_id=adapter_id)
        if item['update']:
            train=tmp_path/(item['key']+'-train');train.mkdir();(train/'run').mkdir();adapter.rename(train/'run/adapter')
            item['source_training']=str(train);c.update(training_root=str(train),adapter=str(train/'run/adapter'))
            atomic_json(train/'config.json',dict(step=item['update']-1,arm=item['arm'],seed=317,lr=1e-4))
            atomic_json(train/'launch.json',dict(image='image',container_id='training',config_sha256=file_hash(train/'config.json')))
            atomic_json(gate/'verified.json',dict(updates=item['update'],arm=item['arm'],root=str(train),suite=str(gate),adapter_id=adapter_id))
            atomic_json(gate/'ACCEPTED.json',{})
        atomic_json(gate/'config.json',c);atomic_json(gate/'launch.json',dict(image='image',container_id='gate'))
        atomic_json(gate/'source-manifest.json',{});atomic_json(gate/'run/READY.json',{})
    path=tmp_path/'rule.json';atomic_json(path,value);cfg['rule']=str(path);cfg['pins'][str(path)]=file_hash(path)
    # Numerical gate internals have their own tests; this verifies the real
    # selection/lineage/export binding with only external acceptance stubbed.
    monkeypatch.setattr(module,'completed',lambda *a:True)
    monkeypatch.setattr(module,'verify_gate',lambda *a:None)
    monkeypatch.setattr(module,'verify_final',lambda *a:None)
    result=module.resolve(cfg)
    assert [r['key'] for r in result['checkpoints']]==[r['key'] for r in value['checkpoints']]
    assert len(result['resolved_sources'])==7 and result['target_feedback_used'] is False
    final=Path(result['resolved_sources'][-1]['adapter'])/'adapter_model.safetensors';final.write_bytes(b'changed')
    with pytest.raises(ValueError,match='content changed'):module.resolve(cfg)
