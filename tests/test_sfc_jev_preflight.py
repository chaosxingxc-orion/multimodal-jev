import copy
import pytest
from test_sfc_jev import load_tool, record
from multimodal_jev.sfc_consumption import split_record
from multimodal_jev.sfc_jev import request_from_native


def requests():
    rows=[]
    for level in ('level_1','level_2','level_3_1','level_3_2'):
        for group in ('id_1','id_2'):
            for stage in ((1,2) if level.startswith('level_3') else (1,)):
                row=record(stage=stage)
                # Build ordinary source rows first, then change input-side identities.
                native,_=split_record(row)
                old=native['source_group'];new=old.replace('level_3_1',level).replace('id_1',group)
                native['source_group']=new
                native['id']=native['id'].replace('level_3_1',level).replace('id_1',group)
                native['input']['audios']=[p.replace('level_3_1',level).replace('id_1',group) for p in native['input']['audios']]
                if not level.startswith('level_3'):
                    native['id']=native['id'].replace('_1.wav','.wav')
                    native['input']['audios']=[p.replace('_1.wav','.wav') for p in native['input']['audios']]
                rows.append(request_from_native(native))
    return rows


def test_preflight_uses_all_decisions_of_four_complete_train_groups():
    tool=load_tool('preflight_sfc_jev_omni');rows=requests();selected=tool.select(rows)
    assert len(selected)==6 and len({r['source_group'] for r in selected})==4
    assert all(r['source_group'].endswith('/id_1') for r in selected)
    assert sum(len(r['input']['decisions']) for r in selected)==36
    assert tool.select(rows[::-1])==selected[::-1]


@pytest.mark.parametrize('change',['missing_turn','duplicate','test','wrong_stage','missing_level'])
def test_preflight_rejects_incomplete_or_wrong_identity(change):
    tool=load_tool('preflight_sfc_jev_omni');rows=requests()
    if change=='missing_turn':rows=[r for r in rows if not (r['source_group']=='level_3_1/train_id/id_1' and len(r['input']['context']['audios'])==2)]
    if change=='duplicate':rows.append(copy.deepcopy(rows[0]))
    if change=='test':rows[0]['split']='test_id'
    if change=='wrong_stage':rows[0]['source_group']='level_1/train_id/id_0'
    if change=='missing_level':rows=[r for r in rows if not r['source_group'].startswith('level_2')]
    with pytest.raises(ValueError):tool.select(rows)


def test_non_hop_aligned_waveform_coverage():
    tool=load_tool('preflight_sfc_jev_omni')
    assert tool.expected_frames([16001,32001],160)==[101,200]
    assert tool.expected_frames([16001],160)==[100]
