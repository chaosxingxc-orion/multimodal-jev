from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from preflight_sfc_omni import select


def fixture():
    return [dict(example_id=f'{level}/{group}/{stage}',
        metadata=dict(official_split='train_id',level=level,source_group=f'{level}/{group}'),
        input={'audios':['audio']*stage})
        for level in ('level_1','level_2','level_3_1','level_3_2') for group in ('b','a')
        for stage in (range(1,3) if level.startswith('level_3') else [1])]


def test_preflight_selects_complete_conversations_without_gold():
    rows=fixture();selected=select(rows)
    assert len(selected)==6
    assert all('/a/' in r['example_id'] for r in selected)
    assert {r['example_id'] for r in select(rows[::-1])}=={r['example_id'] for r in selected}


def test_test_splits_and_lost_second_turn_are_rejected():
    rows=fixture();rows[0]['metadata']['official_split']='test_id'
    with pytest.raises(ValueError,match='TRAIN'):select(rows)
    rows=[r for r in fixture() if r['example_id']!='level_3_1/a/2']
    with pytest.raises(ValueError,match='complete'):select(rows)
