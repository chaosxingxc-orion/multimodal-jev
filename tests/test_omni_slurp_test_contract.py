import json
import pytest
from multimodal_jev import omni_slurp_test_contract as module
from multimodal_jev.baseline import file_hash


def test_complete_native_test_and_fixed_baseline(tmp_path,monkeypatch):
    rows=[dict(id=str(i),source_group=str(i%2974),split='test') for i in range(13078)]
    requests=tmp_path/'test.requests.jsonl';requests.write_text('\n'.join(map(json.dumps,rows)))
    taxonomy=tmp_path/'taxonomy.json';taxonomy.write_text('{}')
    instruction=tmp_path/'prompt.txt';instruction.write_text('frozen composite prompt')
    cfg=dict(split='test',task_format='slurp-native',evaluation_phase='slurp-native-test',
        audio_condition='actual',parameters_frozen=True,max_new_tokens=256,max_context_tokens=4096,
        dtype='bfloat16',seed=17,sampling={'temperature':0.,'n':1},route='audio-direct',audio_model='/model',
        requests=str(requests),instruction=str(instruction),taxonomy=str(taxonomy),taxonomy_sha256=file_hash(taxonomy))
    prior=tmp_path/'prior.json';prior.write_text(json.dumps(cfg))
    cfg.update(population_reference=str(prior),population_reference_sha256=file_hash(prior))
    monkeypatch.setattr(module,'REQUEST_SHA',file_hash(requests))
    (tmp_path/'MANIFEST.json').write_text(json.dumps(dict(release_id=module.RELEASE,
        identity={'taxonomy_sha256':file_hash(taxonomy)},splits={'test':dict(requests_sha256=module.REQUEST_SHA,targets_sha256=module.TARGET_SHA,recordings=13078)})))
    assert module.select_native_test(cfg,rows)==rows
    for changed in [rows[:-1],rows[:-1]+[rows[0]],rows[:-1]+[{**rows[-1],'split':'train'}]]:
        with pytest.raises(ValueError,match='all13078'):module.select_native_test(cfg,changed)
    for change in [dict(seed=99),dict(max_new_tokens=128),dict(source_groups=['0']),dict(split='train')]:
        with pytest.raises(ValueError):module.select_native_test({**cfg,**change},rows)
