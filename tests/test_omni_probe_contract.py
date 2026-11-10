import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.omni_probe_contract import select_learning


def setup(tmp_path):
    groups=['g'+str(i) for i in range(64)]
    p=tmp_path/'partition.json';p.write_text(json.dumps(dict(learning=groups,selection=['heldout'])))
    cfg=dict(split='train',audio_condition='actual',source_groups=groups,partition=str(p),partition_sha256=file_hash(p))
    rows=[dict(id=g+':'+str(v),source_group=g,split='train') for g in groups for v in range(6)]
    return cfg,rows


def test_fixed_population_rejects_selection_substitution_and_missing_views(tmp_path):
    cfg,rows=setup(tmp_path)
    assert select_learning(cfg,rows)==rows
    for change in [dict(split='test'),dict(audio_condition='transcript'),dict(source_groups=list(reversed(cfg['source_groups']))),
                   dict(source_groups=cfg['source_groups'][:-1]+['heldout'])]:
        with pytest.raises(ValueError):select_learning({**cfg,**change},rows)
    for altered in [rows[:-1],rows[:-1]+[rows[0]]]:
        with pytest.raises(ValueError):select_learning(cfg,altered)
    assert select_learning({**cfg,'audio_condition':'zeros'},rows)==rows
    assert len(select_learning({**cfg,'source_groups':cfg['source_groups'][:1]},rows))==6
    with open(cfg['partition'],'a') as f:f.write(' ')
    with pytest.raises(ValueError,match='partition changed'):select_learning(cfg,rows)
