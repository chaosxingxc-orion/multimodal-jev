import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.omni_selection_contract import actor_config, select_registered


def fixture(tmp_path):
    partition = dict(learning=['l'+str(i) for i in range(64)],
                     selection=['s'+str(i) for i in range(256)])
    rows = [dict(id=g+':'+str(v), source_group=g, split='train')
            for g in partition['learning']+partition['selection'] for v in range(6)]
    paths = {k:tmp_path/(k+'.json') for k in ('requests','partition','asset_audit','instruction','selection_protocol')}
    paths['requests'].write_text('\n'.join(json.dumps(r) for r in rows))
    paths['partition'].write_text(json.dumps(partition))
    paths['asset_audit'].write_text('{}')
    paths['instruction'].write_text('listen')
    cfg = dict(split='train', audio_condition='actual', parameters_frozen=True,
               evaluation_phase='selection', source_groups=partition['selection'],
               partition_sha256=file_hash(paths['partition']), seed=17,
               **{k:str(p) for k,p in paths.items()})
    protocol = dict(schema='jev-omni-selection-protocol-v1', actor_config=actor_config(cfg),
                    prompts={'parent':file_hash(paths['instruction']), 'p1':'candidate-hash'},
                    **{k+'_sha256':file_hash(paths[k]) for k in ('requests','partition','asset_audit')})
    paths['selection_protocol'].write_text(json.dumps(protocol))
    cfg['selection_protocol_sha256'] = file_hash(paths['selection_protocol'])
    return cfg, rows


def test_exact_selection_and_no_learning_or_test_substitution(tmp_path):
    cfg, rows = fixture(tmp_path)
    selected = select_registered(cfg, rows)
    assert len(selected) == 1536
    assert all(r['source_group'].startswith('s') for r in selected)
    for change in [dict(split='test'), dict(audio_condition='zeros'),
                   dict(source_groups=cfg['source_groups'][:-1]+['l0']),
                   dict(seed=42), dict(parameters_frozen=False), dict(evaluation_phase='learning')]:
        with pytest.raises(ValueError):
            select_registered({**cfg, **change}, rows)
    for changed in [rows[:-1], rows[:-1]+[rows[-2]],
                    rows[:-1]+[{**rows[-1], 'split':'test'}]]:
        with pytest.raises(ValueError, match='complete unique'):
            select_registered(cfg, changed)


@pytest.mark.parametrize('key', ['requests','partition','asset_audit','instruction','selection_protocol'])
def test_mutated_inputs_and_unregistered_prompt_rejected(tmp_path, key):
    cfg, rows = fixture(tmp_path)
    with open(cfg[key], 'a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError):
        select_registered(cfg, rows)
