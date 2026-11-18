from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from audit_native_baseline_uncertainty import cluster_interval,summarize


def fixture():
    return [dict(id=str(i),source_group=g,speaker_id='s'+str(i%2),gold_intent=c,
                 correct=y,malformed=not y) for i,(g,c,y) in enumerate([
        ('a','common',True),('a','common',True),('a','common',False),('b','rare',False)])]


def test_whole_group_sampling_matches_scalar_resample_and_weights_recordings():
    records=fixture();r=cluster_interval(records,'source_group','correct',replicates=100)
    groups={'a':[True,True,False],'b':[False]};rng=np.random.default_rng(317);values=[]
    for _ in range(100):
        draw=rng.integers(0,2,size=2)
        outcomes=[v for index in draw for v in groups[['a','b'][index]]]
        values.append(sum(outcomes)/len(outcomes))
    assert r['estimate']==.5 and r['clusters']==2
    assert [r['lower95'],r['upper95']]==list(np.quantile(values,[.025,.975],method='linear'))
    assert r==cluster_interval(list(reversed(records)),'source_group','correct',replicates=100)


def test_malformed_stays_denominator_and_macro_is_not_micro():
    r=summarize(fixture())
    assert r['examples']==4 and r['joint_correct']==2 and r['malformed']==2
    assert r['macro_recall_over_observed_classes']==pytest.approx(1/3)
    assert r['per_class']['rare']['recall']==0
    assert not r['optimization_allowed']
    with pytest.raises(ValueError,match='duplicate'):
        summarize(fixture()+fixture()[:1])


def test_missing_cluster_and_nonboolean_outcome_rejected():
    records=fixture();records[0]['speaker_id']=''
    with pytest.raises(ValueError,match='cluster'):
        cluster_interval(records,'speaker_id','correct')
    records=fixture();records[0]['correct']=1
    with pytest.raises(ValueError,match='boolean'):
        cluster_interval(records,'source_group','correct')
