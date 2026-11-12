from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import prepare_fsc_transfer_matrix as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path):
    rows=[dict(key=k,adapter_id='same-content') for k in ['step0']+[f'{a}-step{s}' for a in ('D','Q','P') for s in (6,12)]]
    value=dict(resolved_sources=rows,checkpoints=rows,target_feedback_used=False,image='image',base_id='base',runtime_profile='profile')
    for name in ('task.json','population.json','source-selection.json'):atomic_json(tmp_path/name,{})
    return value


def test_complete_protocol_consumable_and_identical_exports_retained(tmp_path):
    value=fixture(tmp_path);result=module.protocol(tmp_path,value,tmp_path/'task.json',tmp_path/'population.json')
    assert len(result['checkpoints'])==7 and len({r['root'] for r in result['checkpoints']})==7
    assert result['source_selection_sha256']==file_hash(tmp_path/'source-selection.json')
    assert result['selection_rule']=='source_only_frozen_before_target_inference'


@pytest.mark.parametrize('bad',['missing','feedback','mismatch'])
def test_invalid_selection_rejected(tmp_path,bad):
    value=fixture(tmp_path)
    if bad=='missing':value['resolved_sources']=value['resolved_sources'][:-1]
    elif bad=='feedback':value['target_feedback_used']=True
    else:value['checkpoints']=[]
    with pytest.raises(ValueError):module.protocol(tmp_path,value,tmp_path/'task.json',tmp_path/'population.json')


def test_unfinished_sources_create_no_target_artifacts(tmp_path,monkeypatch):
    config=dict(pins={})
    for key in ('resolution_config','task','population'):
        path=tmp_path/(key+'.json');atomic_json(path,{});config[key]=str(path);config['pins'][str(path)]=file_hash(path)
    monkeypatch.setattr(module,'resolve',lambda c:(_ for _ in ()).throw(ValueError('source incomplete')))
    monkeypatch.setattr(module,'audit_inputs',lambda *a:pytest.fail('target inputs consumed prematurely'))
    root=tmp_path/'matrix'
    with pytest.raises(ValueError,match='source incomplete'):module.prepare(config,root)
    assert not root.exists()


def test_full_preparation_binds_all_seven_gates_without_launch(tmp_path,monkeypatch):
    selection=fixture(tmp_path);model=tmp_path/'model';model.mkdir()
    (model/'model.safetensors').write_bytes(b'synthetic model identity only')
    adapter=tmp_path/'adapter';adapter.mkdir();(adapter/'adapter_model.safetensors').write_bytes(b'synthetic adapter identity only')
    selection['base_id']=module.checkpoint_identity(model)[0]
    adapter_id=module.checkpoint_identity(adapter)[0]
    selection['resolved_sources']=[dict(r,adapter_id=adapter_id,adapter=str(adapter)) for r in selection['resolved_sources']]
    selection['checkpoints']=[{k:r[k] for k in ('key','adapter_id')} for r in selection['resolved_sources']]
    resolution=tmp_path/'resolution.json';atomic_json(resolution,{})
    processor=tmp_path/'processor';(processor/'run').mkdir(parents=True);atomic_json(processor/'run/result.json',{})
    template=tmp_path/'template';template.mkdir()
    atomic_json(template/'config.json',dict(model=str(model),base_id=selection['base_id'],runtime_profile='profile',physical_devices=[4],max_model_len=4096))
    atomic_json(template/'source-manifest.json',{})
    argv=['docker','create','--name','template','-v',f'{tmp_path}:{tmp_path}:ro','-v',f'{template}:{template}:rw',
          '-v',f'{template}/source:{template}/source:ro','image','timeout','3600s','python3',str(template/'source/tools/jev/check_slurp_native_span_policy.py'),
          '--config',str(template/'config.json'),'--output',str(template/'run')]
    atomic_json(template/'prepared.json',dict(image='image',physical_devices=[4],create_argv=argv))
    cfg=dict(resolution_config=str(resolution),task=str(tmp_path/'task.json'),population=str(tmp_path/'population.json'),
        processor_root=str(processor),gate_template=str(template),pins={})
    for path in [resolution,tmp_path/'task.json',tmp_path/'population.json',*[template/n for n in ('config.json','prepared.json','source-manifest.json')]]:
        cfg['pins'][str(path)]=file_hash(path)
    monkeypatch.setattr(module,'resolve',lambda c:selection)
    monkeypatch.setattr(module,'audit_inputs',lambda *a:dict(recordings=3793))
    root=tmp_path/'matrix';result=module.prepare(cfg,root)
    assert result['policies']==7 and result['requests_per_policy']==22758
    for value in result['gates']:
        gate=Path(value);c=module.read(gate/'config.json');p=module.read(gate/'prepared.json')
        assert c['adapter_id']==adapter_id and c['population']['recordings']==3793
        for name in ('protocol.json','task.json','population.json','source-selection.json'):
            assert c['pins'][str(root/name)]==file_hash(root/name)
        assert '172800s' in p['create_argv'] and '3600s' not in p['create_argv']
        assert not any((gate/n).exists() for n in ('launch.json','create-attempt.json','run'))
    with pytest.raises(FileExistsError):module.prepare(cfg,root)
