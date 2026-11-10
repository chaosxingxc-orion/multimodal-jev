import importlib.util
import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical

spec=importlib.util.spec_from_file_location('omni_compare',Path(__file__).parents[1]/'tools/jev/compare_omni_waveform.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def make_run(tmp_path,condition):
    root=tmp_path/condition;root.mkdir()
    groups=['g'+str(i) for i in range(64)]
    rows=[dict(id=g+':'+str(i),source_group=g,split='train') for g in groups for i in range(6)]
    requests=tmp_path/'requests.jsonl';requests.write_bytes(b''.join(canonical(r) for r in rows))
    prompt=tmp_path/'prompt.txt';prompt.write_text('listen')
    partition=tmp_path/'partition.json';partition.write_text(json.dumps(dict(learning=groups,selection=['heldout'])))
    cfg=dict(split='train',audio_condition=condition,source_groups=groups,requests=str(requests),instruction=str(prompt),partition=str(partition),partition_sha256=file_hash(partition))
    identity=dict(config=cfg,requests_sha256=file_hash(requests),instruction_sha256=file_hash(prompt))
    pred=[dict(example_id=r['id'],replicate=0,audio_condition=condition) for r in rows]
    (root/'predictions.jsonl').write_bytes(b''.join(canonical(r) for r in pred))
    (root/'identity.json').write_text(json.dumps(identity))
    (root/'selection.json').write_text(json.dumps({'ids':[r['id'] for r in rows]}))
    (root/'terminal.json').write_text('{"exit_code":0}')
    (root/'result.json').write_text(json.dumps(dict(status='passed',weight_updates=0,model_files_unchanged=True,audio_condition=condition,predictions=384,requests=384,predictions_sha256=file_hash(root/'predictions.jsonl'))))
    return root


def test_loader_rejects_duplicate_missing_or_wrong_condition_even_with_new_hash(tmp_path):
    root=make_run(tmp_path,'actual')
    assert len(module.load(root,'actual')[2])==384
    with pytest.raises(ValueError):module.load(root,'zeros')
    p=root/'predictions.jsonl';rows=list(map(json.loads,p.read_text().splitlines()));rows[-1]=rows[0]
    p.write_bytes(b''.join(canonical(r) for r in rows))
    r=json.loads((root/'result.json').read_text());r['predictions_sha256']=file_hash(p);(root/'result.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='unique'):module.load(root,'actual')
