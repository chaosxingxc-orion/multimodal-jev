import copy
import json
from pathlib import Path

import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_jev_full import make_bank
from multimodal_jev.slurp_jev_shards import plan_shards,materialize_shard
from multimodal_jev.omni_slurp_transfer_contract import select_transfer_shard,RUNTIME_KEYS


def fixture(tmp_path):
    release=tmp_path/'release';release.mkdir()
    bank=make_bank(dict(scenario_action_pairs=[['s',str(i)] for i in range(60)],entity_types=[str(i) for i in range(55)]))
    (release/'task-bank.json').write_bytes(canonical(bank));digest=file_hash(release/'task-bank.json')
    bundles=[dict(id=str(i),source_group=str(i%2974),split='test',audio_sha256='hash',input={'audio':'/audio','task_bank_sha256':digest}) for i in range(13078)]
    (release/'test.bundles.jsonl').write_bytes(b''.join(map(canonical,bundles)))
    (release/'MANIFEST.json').write_bytes(canonical(dict(identity={'task_bank_sha256':digest},splits={'test':dict(recordings=13078,source_groups=2974,decisions=2197104,bundles_sha256=file_hash(release/'test.bundles.jsonl'))})))
    plan=plan_shards(release,'test',5);plan_path=tmp_path/'plan.json';plan_path.write_bytes(canonical(plan))
    out=tmp_path/'inputs';receipt=materialize_shard(plan_path,'shard-0000',out)
    instruction=tmp_path/'prompt.txt';instruction.write_text('unchanged parent')
    audit=tmp_path/'audit.json';audit.write_text('{}')
    prior=tmp_path/'selection-protocol.json';prior.write_bytes(canonical({'prompts':{'parent':file_hash(instruction),'p1':'candidate'}}))
    selected=tmp_path/'selection-result.json';selected.write_bytes(canonical(dict(selected_id='parent',protocol_sha256=file_hash(prior),test_feedback_used=False,weight_updates=0,selected_instruction_sha256=file_hash(instruction))))
    cfg=dict(split='test',evaluation_phase='slurp-jev-transfer',task_format='jev',audio_condition='actual',parameters_frozen=True,
        max_new_tokens=128,max_context_tokens=4096,dtype='bfloat16',seed=17,sampling={'temperature':0.,'n':1},
        audio_model='/model',route='audio-direct',model_family='qwen3_omni_moe',instruction=str(instruction),asset_audit=str(audit),
        requests=str(out/'test.requests.jsonl'),source_groups=receipt['source_groups'],shard_id='shard-0000',prompt_id='parent')
    protocol=dict(schema='jev-omni-slurp-transfer-v1',runtime={k:cfg[k] for k in RUNTIME_KEYS},
        slurp_feedback_used_for_prompt_optimization=False,prompts={'parent':file_hash(instruction)},asset_audit_sha256=file_hash(audit),
        selection_result=str(selected),selection_result_sha256=file_hash(selected),selection_protocol=str(prior),selection_protocol_sha256=file_hash(prior),
        shard_plan=str(plan_path),shard_plan_sha256=file_hash(plan_path))
    path=tmp_path/'protocol.json';path.write_bytes(canonical(protocol));cfg.update(transfer_protocol=str(path),transfer_protocol_sha256=file_hash(path))
    return cfg,[json.loads(line) for line in Path(cfg['requests']).open()]


def test_full168_tasks_in_shard_and_no_runtime_or_prompt_substitution(tmp_path):
    cfg,rows=fixture(tmp_path)
    assert len(rows)==840 and select_transfer_shard(cfg,rows)==rows
    for altered in [rows[:-1],rows[:-1]+[rows[0]],list(reversed(rows))]:
        with pytest.raises(ValueError,match='no task filtering'):select_transfer_shard(cfg,altered)
    for change in [dict(max_new_tokens=256),dict(seed=42),dict(prompt_id='p1'),dict(split='train')]:
        with pytest.raises(ValueError):select_transfer_shard({**cfg,**change},rows)
    Path(cfg['instruction']).write_text('modified parent')
    with pytest.raises(ValueError,match='unchanged prompt'):select_transfer_shard(cfg,rows)


def test_materialization_is_exclusive_and_target_free(tmp_path):
    cfg,rows=fixture(tmp_path)
    assert not list((tmp_path/'release').glob('*targets*'))
    plan=tmp_path/'plan.json'
    with pytest.raises(FileExistsError):materialize_shard(plan,'shard-0000',tmp_path/'inputs')
    manifest=Path(cfg['requests']).parent/'MANIFEST.json';data=json.loads(manifest.read_text())
    data['recordings']-=1;manifest.write_bytes(canonical(data))
    with pytest.raises(ValueError,match='complete registered'):select_transfer_shard(cfg,rows)


def test_transfer_registration_requires_complete_reproducible_selection(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
    import register_omni_slurp_transfer as registration
    cfg,_=fixture(tmp_path)
    selection=tmp_path/'selection';selection.mkdir()
    (selection/'evaluation-terminal.json').write_text('{"exit_code":0}')
    prior=json.loads((tmp_path/'selection-protocol.json').read_text())
    (selection/'selection-protocol.json').write_bytes(canonical(prior))
    result=json.loads((tmp_path/'selection-result.json').read_text())
    (selection/'selection-result.json').write_bytes(canonical(result))
    (selection/'parent/run').mkdir(parents=True)
    # Current production FSC config relies on the default Jev task format.
    parent_cfg={k:v for k,v in cfg.items() if k!='task_format'}
    (selection/'parent/run/identity.json').write_bytes(canonical(dict(config=parent_cfg,
        asset_audit_sha256=file_hash(cfg['asset_audit']),versions={'vllm':'fixed'},model_files=['fixed'])))
    (selection/'parent/launch.json').write_bytes(canonical({'image':'fixed'}))
    monkeypatch.setattr(registration,'select',lambda *args:result)
    path=tmp_path/'registered.json'
    report=registration.register(selection,tmp_path/'plan.json',path)
    assert report['runtime']['task_format']=='jev'
    assert report['requests_per_prompt']==2197104 and set(report['prompts'])=={'parent'}
    monkeypatch.setattr(registration,'select',lambda *args:{**result,'selected_id':'p1'})
    with pytest.raises(ValueError,match='reproduce'):registration.register(selection,tmp_path/'plan.json',tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()


def test_transfer_shard_preparation_mounts_all_evidence_without_dispatch(tmp_path,monkeypatch):
    import sys,subprocess
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
    import prepare_omni_slurp_transfer_shard as preparation
    cfg,_=fixture(tmp_path)
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'instruction.txt').write_bytes(Path(cfg['instruction']).read_bytes())
    (parent/'config.json').write_bytes(canonical({'asset_audit':cfg['asset_audit']}))
    model=tmp_path/'model';model.mkdir()
    protocol_path=Path(cfg['transfer_protocol']);protocol=json.loads(protocol_path.read_text())
    protocol['runtime']['audio_model']=str(model);protocol['image']='image'
    protocol_path.write_bytes(canonical(protocol))
    native=tmp_path/'native';native.mkdir()
    (native/'evaluation-terminal.json').write_text('{"exit_code":0}')
    argv=['docker','run','--name','native','--device','/dev/davinci4:/dev/davinci4',
        '--device','/dev/davinci5:/dev/davinci5','-e','HCCL_IF_BASE_PORT=62500',
        '-v',str(native)+':'+str(native)+':rw','-v',str(tmp_path)+':'+str(tmp_path)+':ro',
        'image','timeout','--signal=TERM','--kill-after=30s','3600s','python3',
        str(native/'source/tools/jev/infer_qwen3_omni_slurp_probe.py')]
    (native/'launch.json').write_bytes(canonical(dict(argv=argv,image='image',scorer_python='/python',scorer_repository='/scorer')))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    root=tmp_path/'transfer'
    receipt=preparation.prepare(root,protocol_path,'shard-0000','parent',native,commit,[6,7])
    assert receipt['dispatched'] is False and receipt['requests']==840 and receipt['recordings']==5
    intent=json.loads((root/'intent.json').read_text())
    assert intent['physical_devices']==[6,7] and 'HCCL_IF_BASE_PORT=63500' in intent['argv']
    assert '/dev/davinci6:/dev/davinci6' in intent['argv']
    assert '/dev/davinci4:/dev/davinci4' not in intent['argv']
    name=intent['argv'][intent['argv'].index('--name')+1]
    import hashlib
    assert name=='jev-transfer-parent-shard-0000-'+hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    assert (root/'source/tools/jev/infer_qwen3_omni_slurp_transfer.py').exists()
    assert not (root/'launch.json').exists()
