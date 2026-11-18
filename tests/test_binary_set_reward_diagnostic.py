import importlib.util
import itertools
from collections import defaultdict
from pathlib import Path

spec=importlib.util.spec_from_file_location('reward_diagnostic',Path(__file__).parents[1]/'tools/jev/analyze_binary_set_rewards.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_exact_binomial_against_every_binary_action():
    for k in (0,1,2,4):
        q=.3;expected=module.exact_diagnostic(4,k,q,4)
        distributions={name:defaultdict(float) for name in expected['rewards']}
        for action in itertools.product([0,1],repeat=4):
            tp=sum(action[:k]);fp=sum(action[k:]);fn=k-tp;tn=4-k-fp
            reward={'bit_accuracy':(tp+tn)/4,'exact_set':float(action==(1,)*k+(0,)*(4-k)),
                    'set_f1':2*tp/(sum(action)+k) if sum(action)+k else 1.}
            p=q**sum(action)*(1-q)**(4-sum(action))
            for name,r in reward.items():distributions[name][r]+=p
        for name,d in distributions.items():
            assert abs(sum(r*p for r,p in d.items())-expected['rewards'][name]['mean'])<1e-14
            assert abs(sum(p**4 for p in d.values())-expected['rewards'][name]['constant_reward_group_probability'])<1e-14


def test_degenerate_controls_distinguish_coverage_and_negatives():
    x=module.exact_diagnostic(60,1,.5)
    assert x['controls']['return_none']['bit_accuracy']==59/60
    assert x['controls']['return_none']['set_f1']==0
    overlap=module.exact_diagnostic(6,2,.5)
    assert overlap['controls']['oracle_single']['set_f1']==2/3
    assert overlap['controls']['oracle_full']['set_f1']==1
    empty=module.exact_diagnostic(6,0,.5)
    assert empty['controls']['return_none']['set_f1']==1
    assert empty['controls']['return_all']['set_f1']==0
