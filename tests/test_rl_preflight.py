"""Artifact/selection tests; fixture bytes are deliberately not model weights."""
import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.policy_identity import validate_adapter_export
from multimodal_jev.rl_preflight import validate_schedule
from multimodal_jev.task_schema import DecisionSchema,JevExample


def fixture(tmp_path,adapted=False):
    base=tmp_path/'base';base.mkdir()
    (base/'config.json').write_text('{}')
    (base/'model.safetensors').write_bytes(b'fixture base, not model weights')
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture audio, not a recording')
    example=JevExample('ex','train',str(audio),DecisionSchema('s','noul','Question?'),
                       True,{'audio_sha256':file_hash(audio)})
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps(example.to_record())+'\n')
    config={'initial_checkpoint':str(base),'initial_adapter':None,
            'manifest':str(manifest),'batches':[['ex'],['ex']],
            'seed':17,'reward_mode':'grpo','pair_bonus_weight':0.2,
            'sampling':{'temperature':1.,'top_p':1.,'top_k':-1,'n':2,'max_tokens':16},
            'engine':{},'actor':{'model_path':str(base)},
            'export_root':str(tmp_path/'exports')}
    if adapted:
        adapter=tmp_path/'sft'/'adapter';adapter.mkdir(parents=True)
        (adapter/'adapter_config.json').write_text('{}')
        (adapter/'adapter_model.safetensors').write_bytes(b'fixture adapter')
        checkpoint=tmp_path/'sft-export';checkpoint.mkdir()
        (checkpoint/'config.json').write_text('{}')
        (checkpoint/'model.safetensors').write_bytes(b'fixture merged weights')
        report={'status':'passed','training_run':str(adapter.parent),
                'base_sha256':file_hash(base/'model.safetensors'),
                'adapter_files':{p.name:file_hash(p) for p in adapter.iterdir()},
                'export_files':{p.name:file_hash(p) for p in checkpoint.iterdir()}}
        (checkpoint/'EXPORT.json').write_text(json.dumps(report))
        config.update(initial_checkpoint=str(checkpoint),initial_adapter=str(adapter))
    return config


def test_base_and_sft_export_preflight(tmp_path):
    config=fixture(tmp_path,adapted=True)
    report=validate_schedule(config)
    assert report['initial_policy_id']!=report['actor_base_policy_id']
    assert report['unique_examples']==report['unique_audio']==1
    assert report['scheduled_responses']==4
    assert report['initial_adapter_files']['adapter_model.safetensors']==file_hash(
        Path(config['initial_adapter'])/'adapter_model.safetensors')


def test_projection_flag_must_be_explicit_boolean(tmp_path):
    config=fixture(tmp_path)
    config['actor']['response_logits_only']='false'
    with pytest.raises(ValueError,match='boolean'):validate_schedule(config)
    config['actor']['response_logits_only']=True
    assert validate_schedule(config)['unique_examples']==1


@pytest.mark.parametrize('mutation,message',[
    ('adapter','adapter/base'),('base','adapter/base'),
    ('export','published export file changed'),('extra','export file set'),
    ('missing_adapter','initial_adapter'),('failed_export','adapter/base')])
def test_changed_sft_lineage_rejected(tmp_path,mutation,message):
    config=fixture(tmp_path,adapted=True)
    if mutation=='adapter':
        (Path(config['initial_adapter'])/'adapter_model.safetensors').write_bytes(b'changed')
    elif mutation=='base':
        (Path(config['actor']['model_path'])/'model.safetensors').write_bytes(b'changed')
    elif mutation=='export':
        (Path(config['initial_checkpoint'])/'model.safetensors').write_bytes(b'changed')
    elif mutation=='extra':
        (Path(config['initial_checkpoint'])/'tokenizer.json').write_text('{}')
    elif mutation=='missing_adapter':config['initial_adapter']=None
    else:
        p=Path(config['initial_checkpoint'])/'EXPORT.json'
        report=json.loads(p.read_text());report['status']='failed';p.write_text(json.dumps(report))
    with pytest.raises(ValueError,match=message):validate_schedule(config)


def test_parent_export_must_belong_to_exact_training_run(tmp_path):
    config=fixture(tmp_path,adapted=True)
    with pytest.raises(ValueError,match='expected training run'):
        validate_adapter_export(config['actor']['model_path'],config['initial_checkpoint'],
                                config['initial_adapter'],expected_training_run=tmp_path/'wrong')


@pytest.mark.parametrize('mutation,message',[
    ('empty','TRAIN batch'),('duplicate','duplicate selection'),('missing','selection missing'),
    ('valid','TRAIN only'),('audio','audio changed'),('pair','source_group'),
    ('duplicate_manifest','duplicate manifest'),('sampling','temperature')])
def test_bad_schedule_rejected_before_model_loading(tmp_path,mutation,message):
    config=fixture(tmp_path)
    path=Path(config['manifest']);row=json.loads(path.read_text())
    if mutation=='empty':config['batches']=[]
    elif mutation=='duplicate':config['batches']=[['ex','ex']]
    elif mutation=='missing':config['batches']=[['absent']]
    elif mutation=='valid':row['split']='valid';path.write_text(json.dumps(row)+'\n')
    elif mutation=='audio':Path(row['audio']).write_bytes(b'changed')
    elif mutation=='pair':config['reward_mode']='pair-grpo'
    elif mutation=='duplicate_manifest':path.write_text(path.read_text()*2)
    else:config['sampling']['temperature']=0.8
    with pytest.raises(ValueError,match=message):validate_schedule(config)


def test_relocated_identical_base_is_accepted_by_identity(tmp_path):
    config=fixture(tmp_path);base=Path(config['initial_checkpoint'])
    copy=tmp_path/'identical';copy.mkdir()
    for p in base.iterdir():(copy/p.name).write_bytes(p.read_bytes())
    config['initial_checkpoint']=str(copy)
    report=validate_schedule(config)
    assert report['initial_policy_id']==report['actor_base_policy_id']
