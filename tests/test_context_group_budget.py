import itertools
import math
import pytest
from tools.reports.analyze_context_group_budget import at_least_one


def test_at_least_one_matches_explicit_trials():
    p=.13;n=4
    expected=sum(math.prod(p if b else 1-p for b in bits) for bits in itertools.product((0,1),repeat=n) if any(bits))
    assert at_least_one(p,n)==pytest.approx(expected)
    assert at_least_one(0,16)==0
    assert at_least_one(1,16)==1


def test_tiny_probability_not_rounded_to_zero():
    assert at_least_one(1e-20,256)==pytest.approx(256e-20,rel=1e-14,abs=0)
    with pytest.raises(ValueError):at_least_one(.2,0)
