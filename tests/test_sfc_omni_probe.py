import json
from pathlib import Path
import sys
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import infer_sfc_omni_probe as module
from test_sfc_omni_preflight import fixture


def setup(tmp_path):
    rows=fixture()
    for row in rows:row['supervision']={'official_solution':'must never enter generation'}
    path=tmp_path/'train_id.jsonl';path.write_bytes(b''.join(map(canonical,rows)))
    preflight=tmp_path/'preflight';run=preflight/'run';run.mkdir(parents=True)
    identity=dict(release=str(tmp_path),model='/model',source_files={},train_sha256=file_hash(path),
        selected_ids=[r['example_id'] for r in module.select(rows)])
    (run/'identity.json').write_bytes(canonical(identity))
    result=dict(status='passed',identity_sha256=file_hash(run/'identity.json'),
        records=[{'fits_4096_with_1024_output':True} for _ in range(6)])
    (run/'result.json').write_bytes(canonical(result));(run/'terminal.json').write_text('{"exit_code":0}')
    cfg=dict(split='train_id',parameters_frozen=True,route='audio-direct',dtype='bfloat16',seed=17,
        max_context_tokens=4096,max_new_tokens=1024,sampling={'temperature':0.,'n':1},audio_model='/model',
        processor_preflight=str(preflight),processor_preflight_sha256=file_hash(run/'result.json'))
    return cfg,run


def test_exact_preflight_population_and_no_supervision(tmp_path):
    cfg,_=setup(tmp_path);rows,_=module.select_inputs(cfg)
    assert len(rows)==6 and len({r['metadata']['source_group'] for r in rows})==4
    assert all(set(r)=={'example_id','input','metadata'} for r in rows)


@pytest.mark.parametrize('change',[{'split':'test_id'},{'max_new_tokens':128},{'parameters_frozen':False},
                                 {'adapter':'trained'},{'seed':99},{'route':'asr-text'}])
def test_task_or_runtime_drift_rejected(tmp_path,change):
    cfg,_=setup(tmp_path)
    with pytest.raises(ValueError,match='runtime'):module.select_inputs({**cfg,**change})


def test_failed_preflight_and_changed_audio_manifest_rejected(tmp_path):
    cfg,run=setup(tmp_path);(run/'terminal.json').write_text('{"exit_code":1}')
    with pytest.raises(ValueError,match='successful'):module.select_inputs(cfg)
    (run/'terminal.json').write_text('{"exit_code":0}')
    with (tmp_path/'train_id.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='release changed'):module.select_inputs(cfg)
