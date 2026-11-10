from copy import deepcopy
import hashlib
import json
import pytest

from multimodal_jev.sfc_jev import restore_bundle,split_bundle
from multimodal_jev.sfc_consumption import split_record
from multimodal_jev.sfc import source_identity
from multimodal_jev.sfc_partition import partition
from multimodal_jev.sfc_teacher_feedback import build_feedback
from test_sfc_jev_runtime import bundle,predictions


def fixture():
    r,g=bundle();original=restore_bundle(r,g);rs=[];gs=[]
    for name in ('id_1','id_2'):
        for stage in (1,2):
            row=json.loads(json.dumps(original).replace('id_1',name))
            row['input']['messages']=row['input']['messages'][:2*stage-1]
            row['input']['audios']=row['input']['audios'][:stage]
            meta=row['metadata'];meta['relative_audios']=meta['relative_audios'][:stage]
            meta.update(source_identity(meta['relative_audios']))
            meta['audio_sha256']=[hashlib.sha256(p.encode()).hexdigest() for p in row['input']['audios']]
            row['example_id']='sfc:'+meta['relative_audios'][-1]
            rr,gg=split_bundle(*split_record(row));rs.append(rr);gs.append(gg)
    reg=partition(rs,{rs[0]['id']},selection_fraction=.5)
    return rs,gs,reg


def test_learning_feedback_recomputes_strict_errors_without_repair():
    rs,gs,reg=fixture();pp=[p for r,g in zip(rs[:2],gs[:2]) for p in predictions(r,g)]
    pp[0]['raw_text']='```json\ntrue\n```'
    report=build_feedback(rs,reg,rs[:2],gs[:2],pp)
    assert report['counts']['decisions']==12 and report['counts']['format_valid']==11
    assert len(report['all_decision_summary'])==12
    assert any(r['raw_output']=='```json\ntrue\n```' for r in report['representatives'])
    assert report['teacher_has_audio'] is False and report['selection_or_test_feedback'] is False
    assert 'official_solution' not in json.dumps(report)
    assert '/dataset/' not in json.dumps(report)


@pytest.mark.parametrize('change',['selection','missing_turn','missing_decision','binding','partition','test','extra_prediction'])
def test_feedback_rejects_leakage_incomplete_groups_or_changed_identity(change):
    rs,gs,reg=fixture();rr=deepcopy(rs[:2]);gg=deepcopy(gs[:2]);pp=[p for r,g in zip(rr,gg) for p in predictions(r,g)]
    if change=='selection':
        rr=rs[2:];gg=gs[2:];pp=[p for r,g in zip(rr,gg) for p in predictions(r,g)]
    if change=='missing_turn':rr=rr[:1];gg=gg[:1];pp=[p for p in pp if p['example_id']==rr[0]['id']]
    if change=='missing_decision':pp.pop()
    if change=='binding':pp[0]['request_sha256']='0'*64
    if change=='partition':reg['learning']+=reg['selection']
    if change=='test':rr[0]['split']='test_id'
    if change=='extra_prediction':pp.append(dict(pp[0],example_id='unknown'))
    with pytest.raises(ValueError):build_feedback(rs,reg,rr,gg,pp)
