import copy
import json
from pathlib import Path
import sys
import pytest
from multimodal_jev.fsc_external_comparison import compare_native
from multimodal_jev.fsc_published import render_reference


def fixture():
    task=json.loads((Path(__file__).resolve().parents[1]/'configs/tasks/fsc-span-transfer-v1.json').read_text())
    gold=task['commands'][0]['values'];wrong=task['commands'][1]['values']
    roster=[dict(id='fsc:'+str(i),speaker_id='large' if i<9 else 'small',audio_sha256=str(i)) for i in range(10)]
    targets=[dict(id=r['id'],gold=gold) for r in roster]
    predictions=[dict(example_id=str(i),audio_sha256=str(i),raw_prediction=render_reference(gold if i<9 else wrong)) for i in range(10)]
    rows=[dict(id=r['id'],speaker_id=r['speaker_id'],gold=gold,semantic_prediction={'final':gold},final_correct=True) for r in roster]
    cell=dict(records=rows,recordings=10,final_accuracy=1.)
    policies={k:dict(cells={t+'::native':copy.deepcopy(cell) for t in ('direct','structured')}) for k in ['step0']+[f'{a}-step{s}' for a in ('D','Q','P') for s in (6,12)]}
    return task,roster,targets,predictions,policies


def test_native_paired_comparison_preserves_cluster_sizes():
    r=compare_native(*fixture());b=r['reference'];assert b['accuracy']==.9 and b['accuracy_percentile95']==[0,1]
    assert b['observed_classes']==1 and b['macro_f1_full_ontology']==pytest.approx((18/19)/31)
    assert len(r['native_comparisons'])==14
    for c in r['native_comparisons'].values():
        assert c['jev_only_correct']==1 and c['reference_only_correct']==0 and c['both_correct']==9
        assert c['accuracy_difference_jev_minus_reference']==pytest.approx(.1) and c['difference_percentile95']==[0,1]
    assert r==compare_native(*fixture())


def test_invalid_external_outputs_are_wrong_not_dropped():
    args=list(fixture());args[3][0]['raw_prediction']='invalid'
    args[3][1]['raw_prediction']=render_reference(dict(action='unseen',object='unseen',location='none'))
    r=compare_native(*args);assert r['reference']['recordings']==10 and r['reference']['correct']==7
    assert r['reference']['invalid_or_out_of_ontology']==2


@pytest.mark.parametrize('bad',['matrix','missing','duplicate','audio','gold','speaker','truth','aggregate'])
def test_mismatches_rejected(bad):
    args=list(fixture());policies=args[4];row=policies['step0']['cells']['direct::native']['records'][0]
    if bad=='matrix':policies.pop('P-step12')
    elif bad=='missing':args[3].pop()
    elif bad=='duplicate':args[3].append(args[3][0])
    elif bad=='audio':args[3][0]['audio_sha256']='wrong'
    elif bad=='gold':row['gold']=args[0]['commands'][1]['values']
    elif bad=='speaker':row['speaker_id']='other'
    elif bad=='truth':row['final_correct']=False
    else:policies['step0']['cells']['direct::native']['final_accuracy']=0
    with pytest.raises(ValueError):compare_native(*args)


def test_incomplete_matrix_blocks_reference_and_target_access(monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
    import compare_fsc_external_reference as tool
    def failed(*a):raise ValueError('incomplete matrix')
    monkeypatch.setattr(tool,'score_matrix',failed)
    monkeypatch.setattr(tool,'authenticate_reference',lambda *a:pytest.fail('reference read before complete matrix'))
    monkeypatch.setattr(tool,'read',lambda *a:pytest.fail('reference config read before complete matrix'))
    monkeypatch.setattr(tool,'scoring_targets',lambda *a:pytest.fail('targets before matrix'))
    with pytest.raises(ValueError,match='incomplete'):tool.run('/protocol','hash','/reference')


def test_failed_predecessor_never_scores_reference(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
    import queue_fsc_external_report as queue
    from multimodal_jev.baseline import atomic_json,file_hash
    root=tmp_path/'prior';(root/'run').mkdir(parents=True)
    atomic_json(root/'config.json',{});atomic_json(root/'launch.json',dict(pid=123,argv=['python']))
    atomic_json(root/'run/terminal.json',dict(exit_code=1))
    cfg=dict(predecessor=str(root),pins={str(root/n):file_hash(root/n) for n in ('config.json','launch.json')})
    monkeypatch.setattr(queue,'compare',lambda *a:pytest.fail('failed predecessor scored'))
    with pytest.raises(ValueError,match='matrix failed'):queue.run(cfg,tmp_path)
