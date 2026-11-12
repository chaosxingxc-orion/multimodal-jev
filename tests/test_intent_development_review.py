from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from prepare_intent_development_review import select_sources


def rows():return [dict(id=str(i),source_group=str(i//2),role='selection',official_split='train') for i in range(20)]


def test_selection_is_source_uniform_and_order_independent():
    a=select_sources(rows(),{'learning'},budget=5,seed=941)
    assert a==select_sources(list(reversed(rows())),{'learning'},budget=5,seed=941)
    assert len({r['source_group'] for r in a})==5
    assert a!=select_sources(rows(),{'learning'},budget=5,seed=942)


def test_overlap_outside_selected_subset_still_rejected():
    with pytest.raises(ValueError,match='overlaps'):select_sources(rows(),{'9'},budget=1,seed=941)
    bad=rows();bad[-1]['official_split']='test'
    with pytest.raises(ValueError,match='official TRAIN'):select_sources(bad,set(),budget=1,seed=941)
