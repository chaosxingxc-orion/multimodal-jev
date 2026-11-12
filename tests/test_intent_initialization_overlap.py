import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from audit_intent_initialization_overlap import training_population,union_records
from multimodal_jev.span_balance import balanced_binary_batches


def fixture():
    bundles=[dict(id=str(i),targets=[bool(i%2)],supervised=[True]) for i in range(6)]
    _,balance=balanced_binary_batches(bundles,17)
    history=[dict(step=i+1,recordings=ids) for i,ids in enumerate(balance['batch_recording_ids'])]
    return bundles,balance,history,dict(updates=3,recordings=6,training_recordings=6)


def test_actual_history_matches_reconstructed_schedule():
    args=fixture();assert len(training_population(*args))==6


@pytest.mark.parametrize('bad',['schedule','history','step','count','missing'])
def test_rejects_false_training_exposure(bad):
    bundles,balance,history,result=fixture()
    if bad=='schedule':balance['batch_recording_ids'][0].reverse()
    elif bad=='history':history=copy.deepcopy(history);history[0]['recordings']=['wrong','wrong2']
    elif bad=='step':history[0]['step']=0
    elif bad=='count':result['training_recordings']=5
    else:history.pop()
    with pytest.raises(ValueError):training_population(bundles,balance,history,result)


def test_task_suffix_does_not_double_count_same_training_recording():
    a=dict(id='a',file='a.flac',source_group='g',audio_sha256='h')
    b=dict(a,id='a::intent');c=dict(a,id='c',file='c.flac')
    assert len(union_records([a],[b,c]))==2
