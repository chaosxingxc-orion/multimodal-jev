import copy,json,math
import pytest
from test_fsc_span_transfer import config
from multimodal_jev.fsc_span_transfer import compile_request
from multimodal_jev.fsc_transfer_metrics import score_policy,compare_checkpoints,speaker_bootstrap,TEMPLATES,VARIANTS


def fixture():
    task=config();gold=task['commands'][0]['values']
    roster=[dict(id='a',audio='/audio.wav',audio_sha256='hash',speaker_id='speaker')];targets=[dict(id='a',gold=gold)];records=[]
    for template in TEMPLATES:
        for variant in VARIANTS:
            request,binding=compile_request('/audio.wav',task,template,variant=variant,seed=317);probs=[]
            for q in request.questions:
                value=gold if q.id=='final' else gold[q.id]
                y=next(i for i,k in enumerate(q.values) if binding[q.id][k]==value)
                probs.append([math.log(.9 if i==y else .1/(len(q.values)-1)) for i in range(len(q.values))])
            records.append(dict(id='a::'+template+'::'+variant,policy_id='old',audio_sha256='hash',request_identity={'request':json.loads(json.dumps(request.model_input()))},candidate_logprobs=probs))
    return task,roster,targets,records


def test_semantic_rebinding_and_matrix_attribution():
    task,roster,targets,records=fixture();before=score_policy(task,roster,targets,records,policy_id='old')
    assert all(c['final_accuracy']==1 for c in before['cells'].values())
    assert all(r['both_correct']==1 for r in before['robustness'].values())
    after_records=copy.deepcopy(records)
    for r in after_records:
        r['policy_id']='new'
        if '::direct::' in r['id']:
            p=r['candidate_logprobs'][-1];y=max(range(len(p)),key=lambda i:p[i]);j=(y+1)%len(p);p[y],p[j]=p[j],p[y]
    after=score_policy(task,roster,targets,after_records,policy_id='new');comparison=compare_checkpoints(before,after)
    assert comparison['changes']['direct::native']['right_to_wrong']==1
    assert comparison['interaction']['native']['rl_template_interaction']==1
    assert after['robustness']['direct::rebind']['both_correct']==0


def test_correct_components_never_repair_final():
    task,roster,targets,records=fixture()
    for r in records:
        p=r['candidate_logprobs'][-1];y=max(range(len(p)),key=lambda i:p[i]);j=(y+1)%len(p);p[y],p[j]=p[j],p[y]
    result=score_policy(task,roster,targets,records,policy_id='old')['cells']['structured::native']
    assert result['final_accuracy']==0 and result['compatibility']==0
    assert set(result['process_component_accuracy'].values())=={1.}


@pytest.mark.parametrize('bad',['missing','duplicate','policy','audio','request','probability','gold'])
def test_refuses_population_or_identity_corruption(bad):
    t,r,g,p=fixture()
    if bad=='missing':p.pop()
    if bad=='duplicate':p.append(p[0])
    if bad=='policy':p[0]['policy_id']='other'
    if bad=='audio':p[0]['audio_sha256']='other'
    if bad=='request':p[0]['request_identity']['request']['requirements']=['other']
    if bad=='probability':p[0]['candidate_logprobs'][-1][0]=float('nan')
    if bad=='gold':g=[]
    with pytest.raises(ValueError):score_policy(t,r,g,p,policy_id='old')


def test_cross_checkpoint_configuration_change_is_rejected():
    t,r,g,p=fixture();a=score_policy(t,r,g,p,policy_id='old');b=copy.deepcopy(a)
    b['task_sha256']='changed'
    with pytest.raises(ValueError,match='frozen task'):compare_checkpoints(a,b)


def test_cluster_bootstrap_preserves_pairing_and_refuses_one_speaker():
    t,r,g,p=fixture();a=score_policy(t,r,g,p,policy_id='old')
    with pytest.raises(ValueError,match='two speaker'):speaker_bootstrap(a,a,replicates=100)
    for cell in a['cells'].values():
        row=copy.deepcopy(cell['records'][0]);row.update(id='b',speaker_id='second');cell['records'].append(row)
        cell['recordings']=2
    result=speaker_bootstrap(a,a,replicates=100)
    assert result['speaker_clusters']==2
    assert all(v==[0.,0.] for v in result['accuracy_delta_intervals'].values())
    assert result==speaker_bootstrap(a,a,replicates=100)


def test_paired_comparison_rejects_stale_aggregate():
    t,r,g,p=fixture();a=score_policy(t,r,g,p,policy_id='old');b=copy.deepcopy(a)
    b['cells']['direct::native']['final_accuracy']=0.
    with pytest.raises(ValueError,match='aggregate'):compare_checkpoints(a,b)
