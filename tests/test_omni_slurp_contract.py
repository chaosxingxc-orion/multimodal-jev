import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.omni_slurp_contract import select_native_probe, validate_runtime


def test_original_native_population_and_complete_recording_coverage(tmp_path):
    groups=['g'+str(i) for i in range(16)]
    rows=[dict(id=str(i),source_group=groups[i%16],split='train') for i in range(60)]
    requests=tmp_path/'train.requests.jsonl';requests.write_text('\n'.join(map(json.dumps,rows)))
    taxonomy=tmp_path/'taxonomy.json';taxonomy.write_text('{}')
    instruction=tmp_path/'prompt.txt';instruction.write_text('native composite')
    cfg=dict(split='train',task_format='slurp-native',audio_condition='actual',parameters_frozen=True,
        max_new_tokens=256,requests=str(requests),taxonomy=str(taxonomy),taxonomy_sha256=file_hash(taxonomy),
        instruction=str(instruction),source_groups=groups)
    reference=tmp_path/'prior.json';reference.write_text(json.dumps(cfg))
    cfg.update(population_reference=str(reference),population_reference_sha256=file_hash(reference))
    (tmp_path/'MANIFEST.json').write_text(json.dumps(dict(splits={'train':{'requests_sha256':file_hash(requests)}},
        identity={'taxonomy_sha256':file_hash(taxonomy)})))
    assert select_native_probe(cfg,rows)==rows
    for change in [dict(split='test'),dict(audio_condition='zeros'),dict(source_groups=groups[:-1]),dict(max_new_tokens=128)]:
        with pytest.raises(ValueError):select_native_probe({**cfg,**change},rows)
    for changed in [rows[:-1],rows[:-1]+[rows[0]],rows[:-1]+[{**rows[-1],'split':'test'}]]:
        with pytest.raises(ValueError,match='complete'):select_native_probe(cfg,changed)


def test_runtime_config_must_match_implemented_engine():
    cfg=dict(dtype='bfloat16',seed=17,max_context_tokens=4096,max_new_tokens=128,sampling={'temperature':0.,'n':1})
    validate_runtime(cfg)
    validate_runtime({**cfg,'max_new_tokens':256,'task_format':'slurp-native'})
    for change in [dict(seed=42),dict(max_new_tokens=256),dict(dtype='float16'),
                   dict(max_context_tokens=8192),dict(sampling={'temperature':.5,'n':1})]:
        with pytest.raises(ValueError):validate_runtime({**cfg,**change})
