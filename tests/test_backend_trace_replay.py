import copy
import pytest
from tools.reports.context_backend_trace_replay import remote_configuration,accepted
from multimodal_jev.objective_trace import replay
from tests.test_objective_trace import captured
from tests.test_objective_plugin import surrogate


def test_explicit_cpu_backend_preserves_strict_replay(surrogate):
    plugin,_,row,_=captured()
    assert replay(row,plugin,replay_device='cpu')['status']=='objective_formula_replayed'


def test_remote_mapping_preserves_hashes_and_does_not_mutate_config():
    c=dict(objective_plugin=dict(path='/local/plugin.py',sha256='a',options={}),updates=[dict(update=1,
        groups=dict(path='/local/groups.pt',sha256='b'),
        traces=[dict(rank=0,path='/local/trace.json',sha256='c')])])
    before=copy.deepcopy(c);remote,files=remote_configuration(c,'/remote')
    assert c==before and remote['replay_device']=='npu:0'
    assert remote['updates'][0]['traces'][0]['sha256']=='c'
    assert files['evidence/trace-0001-rank-0.json']['path']=='/local/trace.json'


def test_backend_report_cannot_substitute_cpu_or_partial_population():
    c=dict(objective_plugin=dict(sha256='a',options={}),world_size=2,local_batch=64,microbatch=8,
           updates=[dict(update=0)])
    remote=dict(updates=[dict(update=0,remote=True)])
    report=dict(status='finite_objective_traces_replayed',replay_device='npu:0',evidence=remote['updates'],
        objective=dict(api='jev-stage2-objective-v1',sha256='a',options={}),
        updates=[dict(update=0,ranks=2,queries=128,microbatches=16)])
    assert accepted(report,c,remote)['evidence']==c['updates']
    bad=copy.deepcopy(report);bad['replay_device']='cpu'
    with pytest.raises(ValueError):accepted(bad,c,remote)
    bad=copy.deepcopy(report);bad['updates'][0]['queries']=127
    with pytest.raises(ValueError):accepted(bad,c,remote)
