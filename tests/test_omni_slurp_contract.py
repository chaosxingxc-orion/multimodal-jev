import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.omni_slurp_contract import select_native_probe, validate_runtime
from multimodal_jev.omni_slurp_contract import select_jev_probe
from multimodal_jev.slurp_jev_full import make_bank, expand_bundle
from multimodal_jev.consumption import canonical


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


def test_full_jev_probe_rebuilds_every_question_without_targets(tmp_path):
    release=tmp_path/'release';release.mkdir()
    bank=make_bank(dict(scenario_action_pairs=[['s',str(i)] for i in range(60)],
                        entity_types=['entity'+str(i) for i in range(55)]))
    (release/'task-bank.json').write_bytes(canonical(bank))
    bank=json.loads((release/'task-bank.json').read_text())
    bundles=[dict(id='recording'+str(i),split='train',source_group='slurp:12507',audio_sha256='hash',
                  input={'audio':'/audio'+str(i),'task_bank_sha256':file_hash(release/'task-bank.json')}) for i in range(2)]
    (release/'train.bundles.jsonl').write_bytes(b''.join(canonical(b) for b in bundles))
    (release/'MANIFEST.json').write_bytes(canonical(dict(identity={'task_bank_sha256':file_hash(release/'task-bank.json')},
        splits={'train':{'bundles_sha256':file_hash(release/'train.bundles.jsonl')}})))
    rows=[r for b in bundles for r in expand_bundle(b,bank)]
    requests=tmp_path/'train.requests.jsonl';requests.write_bytes(b''.join(canonical(r) for r in rows))
    (tmp_path/'MANIFEST.json').write_bytes(canonical(dict(source_release=str(release),
        requests_sha256=file_hash(requests),source_manifest_sha256=file_hash(release/'MANIFEST.json'))))
    instruction=tmp_path/'prompt.txt';instruction.write_text('all values')
    cfg=dict(split='train',task_format='jev',audio_condition='actual',parameters_frozen=True,
        evaluation_phase='slurp-jev-probe',max_new_tokens=256,requests=str(requests),
        instruction=str(instruction),source_groups=['slurp:12507'])
    reference=tmp_path/'reference.json';reference.write_bytes(canonical(cfg))
    cfg.update(population_reference=str(reference),population_reference_sha256=file_hash(reference))
    assert len(select_jev_probe(cfg,rows))==336
    for changed in [rows[:-1],rows[:-1]+[rows[0]],list(reversed(rows))]:
        with pytest.raises(ValueError,match='all168'):select_jev_probe(cfg,changed)
    altered=[dict(r) for r in rows]
    altered[0]={**altered[0],'input':{**altered[0]['input'],'schema':{**altered[0]['input']['schema'],'question':'changed'}}}
    with pytest.raises(ValueError,match='all168'):select_jev_probe(cfg,altered)
