from pathlib import Path
import sys
from collections import Counter
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from build_semantic_actor_menu import compile_panels

def test_complete_65_55_coverage_without_padding_duplicates():
    i=[f'i{k}' for k in range(65)];s=[f's{k}' for k in range(55)]
    panels=compile_panels(i,s)
    assert len(panels)==17
    assert panels==compile_panels(i[::-1],s[::-1])
    assert Counter(q['definition_id'] for p in panels for q in p['process_questions'])==Counter(i+s)
    assert panels[-1]['configured_counts']=={'I_present':1,'S_present':0}
    assert panels[-2]['configured_counts']=={'I_present':4,'S_present':0}
    for p in panels:
        assert p['arms']['Q_input']==p['arms']['Q_credit']==p['process_questions']
        assert p['arms']['Direct']==[]

def test_incomplete_panels_preserve_full_sweep_objective():
    panels=compile_panels([f'i{k}' for k in range(65)],[f's{k}' for k in range(55)])
    # Deliberately unequal losses and masks, so an incorrect mean-of-panel-means fails.
    values={q['definition_id']:(int(q['definition_id'][1:])+1)/7 for p in panels for q in p['process_questions']}
    masks={k:int(k[1:])%3!=0 for k in values}
    final=[(n+2)/3 for n in range(17)]
    whole=sum(final)/17 + sum(.5/(65 if k[0]=='i' else 55)*v*masks[k] for k,v in values.items())
    streamed=sum(final[n]+sum(17*.5/(65 if q['operator']=='I_present' else 55)*values[q['definition_id']]*masks[q['definition_id']] for q in p['process_questions']) for n,p in enumerate(panels))/17
    assert streamed==pytest.approx(whole)

def test_invalid_support_rejected():
    with pytest.raises(ValueError,match='unique'):compile_panels(['a','a'],['b'])
    with pytest.raises(ValueError,match='width'):compile_panels(['a'],['b'],0)
