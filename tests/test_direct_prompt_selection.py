import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('direct_select',
    Path(__file__).parents[1]/'tools/jev/select_direct_prompt.py')
# CLI sibling module is intentionally imported as a normal script dependency.
import sys
import math
from fractions import Fraction
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'tools/jev'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_tail_handles_full_test_discordance_without_overflow():
    for fixed, regressed in [(0,0),(1,5),(20,31),(100,101)]:
        n=fixed+regressed;k=min(fixed,regressed)
        expected=min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
        assert module.mcnemar_p(fixed,regressed)==pytest.approx(expected)
    expected=float(Fraction(2*sum(math.comb(3793,i) for i in range(1894)),2**3793))
    assert module.mcnemar_p(1900,1893)==pytest.approx(expected,abs=1e-10)
    assert module.mcnemar_p(3793,0)==0.


def row(name, all_six, pairs, total, length=100):
    return {'id': name, 'instruction_characters': length,
            'metrics': {'all_six_correct': all_six,
                        'pairs': {'p': {'both_correct': pairs}},
                        'by_variant': {'v': {'correct': total}}}}


def test_paired_discrimination_precedes_marginal_and_parent_wins_tie():
    parent = row('parent', 0, 0, 1000)
    discriminating = row('p1', 0, 1, 100)
    assert min([parent, discriminating], key=module.rank)['id'] == 'p1'
    assert min([parent, row('p2', 0, 0, 1000, 1)], key=module.rank)['id'] == 'parent'
    assert min([row('p1', 1, 4, 6), row('p2', 0, 200, 1000)], key=module.rank)['id'] == 'p1'
    assert min([row('p2', 0, 1, 100, 90), row('p1', 0, 1, 100)], key=module.rank)['id'] == 'p2'


def test_pair_transitions_keep_boolean_types_and_do_not_pool_views():
    requests=[{'id':str(i),'source_group':'g'+str(i//2)} for i in range(4)]
    targets=[{'id':str(i),'target':True,'metadata':{'variant':'a' if i%2==0 else 'b'}} for i in range(4)]
    parent={str(i):{'value':v} for i,v in enumerate([True,True,False,False])}
    candidate={str(i):{'value':v} for i,v in enumerate([False,1,True,True])}
    report=module.paired_outcomes(requests,targets,parent,candidate)
    assert report['by_view']['a']['fixed']==1 and report['by_view']['a']['regressed']==1
    assert report['by_view']['b']['fixed']==1 and report['by_view']['b']['regressed']==1
    assert report['all_six_by_source_group']['fixed']==1
    assert report['all_six_by_source_group']['regressed']==1
    assert report['all_six_by_source_group']['mcnemar_exact_two_sided_p']==1.
