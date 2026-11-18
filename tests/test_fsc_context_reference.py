import copy
import json
from pathlib import Path

import pytest

from multimodal_jev.fsc_published import render_reference
from tools.reports import compare_fsc_context_reference as tool


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


def fixture():
    task=tool.read('configs/tasks/fsc-span-transfer-v1.json')
    commands=task['commands'];gold=commands[0];wrong=commands[1]
    roster=[dict(id='fsc:'+str(i),speaker_id='large' if i<9 else 'small',audio_sha256=str(i)) for i in range(10)]
    targets=[dict(id=r['id'],gold=gold['values']) for r in roster]
    reference=[dict(example_id=str(i),audio_sha256=str(i),
        raw_prediction=render_reference(gold['values'] if i<9 else wrong['values'])) for i in range(10)]
    rows=[dict(id=r['id'],speaker_id=r['speaker_id'],gold=gold['id'],top1=gold['id'],top1_correct=True) for r in roster]
    policy=dict(policy_id='frozen',records=rows,metrics=dict(queries=10,top1_event_accuracy=1.,
        candidate_ids=[c['id'] for c in commands]))
    return task,roster,targets,reference,{role:copy.deepcopy(policy) for role in ('initial','selected')}


def test_two_policy_native_comparison_and_cluster_weighting():
    result=tool.compare_pair(*fixture());b=result['reference']
    assert b['accuracy']==.9 and b['accuracy_percentile95']==[0.,1.]
    assert b['macro_f1_full_ontology']==pytest.approx(18/19/31)
    assert set(result['native_comparisons'])=={'initial','selected'}
    for c in result['native_comparisons'].values():
        assert c['both_correct']==9 and c['jev_only_correct']==1
        assert c['accuracy_difference_jev_minus_reference']==pytest.approx(.1)
        assert c['difference_percentile95']==[0.,1.]
    assert result==tool.compare_pair(*fixture())


@pytest.mark.parametrize('raw',['invalid',render_reference(dict(action='',object='none',location='none')),
                              render_reference(dict(action='unseen',object='none',location='none'))])
def test_invalid_reference_stays_in_denominator(raw):
    args=list(fixture());args[3][0]['raw_prediction']=raw
    result=tool.compare_pair(*args)['reference']
    assert result['recordings']==10 and result['correct']==8 and result['invalid_or_out_of_ontology']==1


@pytest.mark.parametrize('fault',['policy','reference_missing','reference_duplicate','audio','gold','speaker',
                                'truth','aggregate','duplicate_row','candidate_order','single_speaker'])
def test_pair_mismatches_are_rejected(fault):
    args=list(fixture());row=args[4]['initial']['records'][0]
    if fault=='policy':args[4].pop('selected')
    elif fault=='reference_missing':args[3].pop()
    elif fault=='reference_duplicate':args[3].append(args[3][0])
    elif fault=='audio':args[3][0]['audio_sha256']='wrong'
    elif fault=='gold':row['gold']=args[0]['commands'][1]['id']
    elif fault=='speaker':row['speaker_id']='other'
    elif fault=='truth':row['top1_correct']=1
    elif fault=='aggregate':args[4]['initial']['metrics']['top1_event_accuracy']=.9
    elif fault=='duplicate_row':args[4]['initial']['records'].append(row)
    elif fault=='candidate_order':args[4]['initial']['metrics']['candidate_ids'].reverse()
    else:
        for r in args[1]:r['speaker_id']='same'
    with pytest.raises(ValueError):tool.compare_pair(*args)


@pytest.fixture
def archive(tmp_path,monkeypatch):
    root=tmp_path/'reference/test';root.mkdir(parents=True)
    population=dict(recordings=3793,source_csv_sha256={'test':'csv'},records=[])
    pop=write(tmp_path/'population.json',population);task=write(tmp_path/'task.json',fixture()[0])
    model=tmp_path/'model';model.mkdir();(model/'weights').write_bytes(b'weights')
    identity=dict(config={'split':'test','slu_model':str(model)},csv_sha256='csv',
        models={'slu_model':{'files':[{'file':'weights','sha256':tool.file_hash(model/'weights')}]}})
    write(root/'identity.json',identity);(root/'predictions.jsonl').write_text('')
    result=dict(examples=3793,predictions=3793,missing=0,joint_correct=3774,
        predictions_sha256=tool.file_hash(root/'predictions.jsonl'))
    write(root/'result.json',result);write(root/'terminal.json',dict(exit_code=0))
    state=dict(Status='exited',ExitCode=0,OOMKilled=False,Running=False,Dead=False)
    write(root.parent/'container-terminal.json',dict(id='container',state=state))
    receipt=write(tmp_path/'receipt.json',dict(run=str(root),result=result,
        identity_sha256=tool.file_hash(root/'identity.json'),container_id='container',
        independent_rescore_equal=True,model_files_reverified=True))
    full=dict(passed=True,actual_state=state,reference_container_id='container',
        population_sha256=tool.file_hash(pop),reference_receipt_sha256=tool.file_hash(receipt),
        predictions_sha256=tool.file_hash(root/'predictions.jsonl'),previously_verified_joint_correct=3774,
        alignment=[],speakers=10)
    hist=tmp_path/'historical';write(hist/'result.json',full)
    summary={k:v for k,v in full.items() if k!='alignment'}
    summary.update(root=str(hist),full_result_sha256=tool.file_hash(hist/'result.json'))
    proof=write(tmp_path/'alignment.json',summary)
    spec=dict(receipt=receipt,alignment_receipt=proof,population=pop,task=task)
    spec['pins']={p:tool.file_hash(p) for p in spec.values()}
    monkeypatch.setattr(tool,'align',lambda *a:dict(alignment=[],speakers=10))
    return spec,root,model,hist


def test_archive_rechecks_retained_identity_without_live_docker(archive):
    spec,*_=archive;proof,*_=tool.authenticate_archive(spec)
    assert proof['model_files_reverified'] and proof['audio_files_reverified']
    assert not proof['live_container_inspected'] and proof['recordings']==3793


@pytest.mark.parametrize('fault',['model','predictions','identity','terminal','historical','receipt','population'])
def test_archive_rejects_changed_retained_evidence(archive,fault):
    spec,root,model,hist=archive
    path={'model':model/'weights','predictions':root/'predictions.jsonl','identity':root/'identity.json',
          'terminal':root.parent/'container-terminal.json','historical':hist/'result.json',
          'receipt':Path(spec['receipt']),'population':Path(spec['population'])}[fault]
    path.write_text('{}')
    with pytest.raises((ValueError,KeyError)):tool.authenticate_archive(spec)


@pytest.fixture
def completed(tmp_path,monkeypatch):
    task,roster,targets,predictions,policies=fixture();experiment=tmp_path/'experiment'
    write(experiment/'scores.json',{'old':'score'})
    write(experiment/'status.json',dict(phase='frozen_transfer_collected_and_scored',
        scores_sha256=tool.file_hash(experiment/'scores.json')))
    population=write(tmp_path/'population.json',dict(records=roster));task_path=write(tmp_path/'task.json',task)
    frozen={k:dict(path=p,sha256=tool.file_hash(p)) for k,p in [('population',population),('task',task_path)]}
    contract=write(experiment/'contract.json',frozen)
    write(experiment/'score-spec.json',dict(scoring_contract=dict(path=contract,sha256=tool.file_hash(contract))))
    spec=dict(experiment=str(experiment),pins={},reference=dict(population=population,task=task_path))
    calls=[]
    def authenticate(*a):
        calls.append('archive');return {},dict(records=roster),task,predictions,dict(result=dict(joint_correct=9,examples=10))
    def score(s,out):
        calls.append('actual-runtime');r=dict(runtime_provenance_verified=True,policies=policies,comparison={});write(out,r);return r
    def labels(*a):calls.append('labels');return targets
    monkeypatch.setattr(tool,'authenticate_archive',authenticate)
    monkeypatch.setattr(tool,'reaudited_score',score)
    monkeypatch.setattr(tool,'scoring_targets',labels)
    return spec,calls


def test_completed_pair_is_reaudited_before_label_scoring(completed,tmp_path):
    spec,calls=completed;result=tool.run(spec,tmp_path/'out')
    assert calls==['archive','actual-runtime','labels']
    assert result['reference']['correct']==9 and (tmp_path/'out/comparison.json').is_file()


@pytest.mark.parametrize('fault',['incomplete','score_hash','contract','population','runtime'])
def test_bad_predecessor_cannot_score(completed,tmp_path,monkeypatch,fault):
    spec,calls=completed;root=Path(spec['experiment'])
    if fault=='incomplete':write(root/'status.json',dict(phase='waiting'))
    elif fault=='score_hash':write(root/'scores.json',{})
    elif fault=='contract':write(root/'contract.json',{})
    elif fault=='population':write(Path(spec['reference']['population']),{})
    else:
        def reject(*a):raise ValueError('actual runtime failure')
        monkeypatch.setattr(tool,'reaudited_score',reject)
    with pytest.raises(ValueError):tool.run(spec,tmp_path/'out')
    assert 'labels' not in calls and not (tmp_path/'out/comparison.json').exists()
