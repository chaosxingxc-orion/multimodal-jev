import pytest
from tools.reports.summarize_cpu_audit_profile import summarize


def data():
    return dict(shared=dict(frames=[{'name':'caller'}, {'name':'leaf'}]), profiles=[
        dict(type='sampled', unit='seconds', name='worker', samples=[[0,1], [0,0]], weights=[.2,.1])])


def test_recursive_frames_not_double_counted_and_leaf_separate():
    report = summarize(data())['threads'][0]
    assert report['sampled_seconds'] == pytest.approx(.3)
    assert report['inclusive'][0]['sampled_seconds'] == pytest.approx(.3)
    assert report['leaf'][0]['frame']['name'] == 'leaf'
    assert report['leaf'][0]['sampled_seconds'] == .2


@pytest.mark.parametrize('fault', ['length','index','nan','unit'])
def test_invalid_capture_rejected(fault):
    d=data(); p=d['profiles'][0]
    if fault=='length':p['weights'].pop()
    if fault=='index':p['samples'][0]=[99]
    if fault=='nan':p['weights'][0]=float('nan')
    if fault=='unit':p['unit']='milliseconds'
    with pytest.raises(ValueError):summarize(d)
