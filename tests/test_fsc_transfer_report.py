import copy
import sys
from pathlib import Path
import pytest
from test_fsc_transfer_metrics import fixture
from multimodal_jev.fsc_transfer_metrics import score_policy
from multimodal_jev.fsc_transfer_report import summarize,cell_report,command


def policies():
    task,roster,targets,records=fixture();policy=score_policy(task,roster,targets,records,policy_id='old')
    for cell in policy['cells'].values():
        a=cell['records'][0];b=copy.deepcopy(a);b.update(id='b',speaker_id='second')
        cell['records'].append(b);cell['recordings']=2
    return task,{k:copy.deepcopy(policy) for k in ('step0','D-step6','D-step12')}


def test_perfect_predictions_keep_absent_classes_explicit():
    task,p=policies();result=summarize(task,p,replicates=100)
    c=result['policies']['step0']['structured::native']
    assert c['macro_f1_observed']==1 and c['macro_f1_full_ontology']==pytest.approx(1/31)
    assert c['process_final_counts']=={'00':0,'01':0,'10':0,'11':2}
    assert sum(x['recall'] is None for x in c['per_class'])==30
    assert all(v['percentile95']==[0,0] for row in result['template_gains'].values() for v in row.values())
    assert result==summarize(task,p,replicates=100)


def test_template_gain_without_weight_transfer_is_not_rl_gain():
    task,p=policies()
    for policy in p.values():
        for key,cell in policy['cells'].items():
            if key.startswith('direct'):
                for row in cell['records']:
                    row['semantic_prediction']['final']=task['commands'][1]['values'];row['final_correct']=False
                cell['final_accuracy']=0
    result=summarize(task,p,replicates=100)
    assert all(v['accuracy_gain']==1 and v['percentile95']==[1,1] for row in result['template_gains'].values() for v in row.values())


def test_cluster_draws_preserve_unequal_recording_counts():
    task,p=policies()
    for policy in p.values():
        for key,cell in policy['cells'].items():
            a,b=cell['records'];rows=[]
            for i in range(9):
                row=copy.deepcopy(a);row['id']=f'a{i}';rows.append(row)
            rows.append(b);cell['records']=rows;cell['recordings']=10
            # Structured always right; direct wrong only for large speaker.
            if key.startswith('direct'):
                for row in rows[:-1]:
                    row['semantic_prediction']['final']=task['commands'][1]['values'];row['final_correct']=False
                cell['final_accuracy']=.1
    r=summarize(task,p,replicates=2000)
    for row in r['template_gains'].values():
        for v in row.values():
            assert v['accuracy_gain']==.9 # Not speaker-unweighted .5.
            assert v['percentile95']==[0,1] # Not a narrow independent-row CI.


@pytest.mark.parametrize('bad',['truth','speaker','gold','process'])
def test_semantic_or_population_corruption_rejected(bad):
    task,p=policies();cell=p['step0']['cells']['structured::native'];r=cell['records'][0]
    if bad=='truth':r['final_correct']=False
    elif bad=='speaker':r['speaker_id']=''
    elif bad=='gold':r['gold']={'action':'x'}
    else:r['process_correct']['action']=False
    with pytest.raises(ValueError):cell_report(cell,{command(c['values']) for c in task['commands']})


def test_supplement_cannot_bypass_incomplete_matrix_gate(monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
    import summarize_fsc_transfer_matrix as tool
    def incomplete(*args):raise ValueError('incomplete matrix')
    monkeypatch.setattr(tool,'run',incomplete)
    monkeypatch.setattr(tool,'summarize',lambda *a:pytest.fail('scored before authentication'))
    with pytest.raises(ValueError,match='incomplete'):tool.report('/protocol','hash')
