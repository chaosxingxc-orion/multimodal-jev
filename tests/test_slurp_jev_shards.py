import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_jev_full import make_bank,expand_bundle
from multimodal_jev.slurp_jev_shards import plan_shards,load_shard


def fixture(tmp_path):
    bank=make_bank(dict(scenario_action_pairs=[['s','a'],['s','b']],entity_types=['person']))
    (tmp_path/'task-bank.json').write_bytes(canonical(bank))
    bank=json.loads((tmp_path/'task-bank.json').read_text())
    digest=file_hash(tmp_path/'task-bank.json')
    rows=[dict(id=f'{g}-{i}',split='test',source_group=g,audio_sha256='hash',
        input={'audio':'/audio','task_bank_sha256':digest})
        for g,n in [('b',3),('a',2),('c',5)] for i in range(n)]
    (tmp_path/'test.bundles.jsonl').write_bytes(b''.join(map(canonical,rows)))
    manifest=dict(identity={'task_bank_sha256':digest},splits={'test':dict(recordings=10,
        source_groups=3,decisions=10*len(bank['tasks']),bundles_sha256=file_hash(tmp_path/'test.bundles.jsonl'))})
    (tmp_path/'MANIFEST.json').write_bytes(canonical(manifest))
    return rows,bank


def test_full_population_without_gold_or_splitting_groups(tmp_path):
    rows,bank=fixture(tmp_path)
    plan=plan_shards(tmp_path,'test',4)
    assert plan['recordings']==10 and plan['gold_read'] is False
    assert plan['indivisible_group_exceeds_target'] is True
    assert [s['source_groups'] for s in plan['shards']]==[['a'],['b'],['c']]
    expanded=[]
    for shard in plan['shards']:
        bundles,loaded_bank=load_shard(plan,shard['id'])
        expanded.extend(r for b in bundles for r in expand_bundle(b,loaded_bank))
    expected=[r for b in rows for r in expand_bundle(b,bank)]
    assert {r['id']:r for r in expanded}=={r['id']:r for r in expected}
    assert len(expanded)==len(expected)==60
    plan['shards'][0]['source_groups'].append('b')
    with pytest.raises(ValueError,match='registered shard'):load_shard(plan,'shard-0000')


def test_source_drift_and_incomplete_population_rejected(tmp_path):
    fixture(tmp_path);plan=plan_shards(tmp_path,'test')
    path=tmp_path/'test.bundles.jsonl';path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='source changed'):load_shard(plan,'shard-0000')
    with pytest.raises(ValueError,match='source changed'):plan_shards(tmp_path,'test')
