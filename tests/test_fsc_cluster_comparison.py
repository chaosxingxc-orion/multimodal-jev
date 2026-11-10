from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from multimodal_jev.consumption import canonical
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.fsc_cluster_comparison import compare
from test_six_view_score import fixture as six_views
from test_sfc_jev import load_tool

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def fixture():
    import hashlib
    requests=[];targets=[];left=[];right=[]
    # Two recordings by speaker A improve; one by speaker B regresses.
    for i,speaker in enumerate(('A','A','B')):
        rs,ts,ps=six_views()
        for r,t,p in zip(rs,ts,ps):
            uid=str(i)+':'+t['metadata']['variant'];group='g'+str(i);audio='audio'+str(i)
            r.update(id=uid,source_group=group,audio_sha256=audio)
            t.update(id=uid,request_sha256=hashlib.sha256(canonical(r)).hexdigest())
            t['metadata'].update(source_group=group,audio_sha256=audio,speaker_id=speaker)
            p.update(example_id=uid,request_sha256=t['request_sha256'])
        requests.extend(rs);targets.extend(ts)
        good=deepcopy(ps);bad=deepcopy(ps)
        for p in bad:p['raw_text']='not JSON'
        left.extend(bad if i<2 else good);right.extend(good if i<2 else bad)
    return requests,targets,left,right


def test_speaker_pairing_and_all_six_not_pooled():
    result=compare(*fixture(),draws=2000)
    assert result['requests']==18 and result['source_groups']==3 and result['speakers']==2
    for metric in ('native/correct','pair/noul','all_six','missing_candidate/format_valid'):
        c=result['comparisons'][metric]
        assert c['count']==3 and c['clusters']==2 and c['fixed']==2 and c['regressed']==1
        assert c['delta']==pytest.approx(1/3) and c['delta_ci95']==[-1,1]
    assert len(result['comparisons'])==23


def test_identical_prompts_and_one_speaker_uncertainty():
    rs,ts,left,_=fixture();r=compare(rs,ts,left,left,draws=100)
    assert r['comparisons']['all_six']['delta_ci95']==[0,0]
    for t in ts:t['metadata']['speaker_id']='single'
    r=compare(rs,ts,left,left,draws=100)
    assert r['comparisons']['native/correct']['delta_ci95'] is None


@pytest.mark.parametrize('change',['speaker_missing','speaker_conflict','duplicate','missing','binding','replicate','bool_replicate','draws'])
def test_incomplete_or_bad_cluster_evidence_rejected(change):
    rs,ts,left,right=fixture();kw=dict(draws=100)
    if change=='speaker_missing':ts[0]['metadata'].pop('speaker_id')
    if change=='speaker_conflict':ts[0]['metadata']['speaker_id']='other'
    if change=='duplicate':right.append(right[0])
    if change=='missing':right.pop()
    if change=='binding':right[0]['request_sha256']='changed'
    if change=='replicate':right[0]['replicate']=1
    if change=='bool_replicate':right[0]['replicate']=False
    if change=='draws':kw['draws']=True
    with pytest.raises(ValueError):compare(rs,ts,left,right,**kw)


def test_cli_does_not_read_targets_or_predictions_until_queue_terminal(tmp_path,monkeypatch):
    m=load_tool('compare_fsc_test_clusters');p=tmp_path/'test-protocol.json'
    atomic_json(p,dict(prompts={'parent':'a','p1':'b'},image='image'))
    atomic_json(tmp_path/'evaluation-terminal.json',dict(exit_code=1))
    monkeypatch.setattr(m,'load',lambda *a:pytest.fail('incomplete TEST must not read predictions'))
    with pytest.raises(ValueError,match='complete TEST queue'):m.report(tmp_path,file_hash(p))


def test_exact_failed_container_blocks_target_read(tmp_path,monkeypatch):
    m=load_tool('compare_fsc_test_clusters');p=tmp_path/'test-protocol.json'
    atomic_json(p,dict(prompts={'parent':'a','p1':'b'},image='image'))
    atomic_json(tmp_path/'evaluation-terminal.json',dict(exit_code=0))
    (tmp_path/'parent').mkdir();atomic_json(tmp_path/'parent/launch.json',dict(container_id='cid'))
    monkeypatch.setattr(m,'inspect',lambda _:dict(Id='cid',Image='image',State=dict(Status='running',ExitCode=0)))
    monkeypatch.setattr(m,'load',lambda *a:pytest.fail('failed container must not read predictions'))
    with pytest.raises(ValueError,match='exact successful'):m.report(tmp_path,file_hash(p))
