"""Real source authentication/shards/probabilities; model loading substituted."""
import copy,importlib.util,json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.simple_continuous_resume import authenticate
from multimodal_jev.intent_resident_replay import compare_state
from multimodal_jev.simple_resident_acceptance import compare_updated_batch
from multimodal_jev.resident_service_admission import IMAGE
from test_simple_continuous_resume import fixture as source_fixture


def fixture(tmp_path,monkeypatch):
    config,source,put=source_fixture(tmp_path,monkeypatch);accepted=authenticate(config)
    root=tmp_path/'restore';put(root/'config.json',config)
    put(root/'launch.json',dict(container_id='restore',image=IMAGE,config_sha256=file_hash(root/'config.json')))
    put(root/'terminal.json',dict(exit_code=0));observed=tmp_path/'restore-observed.json'
    put(observed,dict(container_id='restore',image=IMAGE,state=dict(Status='exited',ExitCode=0)))
    ranks=[json.loads((source/'run'/f'loop-rank-{r}/after-0003.json').read_text()) for r in range(4)]
    rollout=json.loads((source/'run/rollout-after-0003.json').read_text())
    sampling=json.loads((source/'run/inputs-0002.json').read_text())['sampling']
    parity=compare_updated_batch(ranks,rollout,sampling,policy_id=accepted['policy_id'],updates=3)
    for rank,state in enumerate(accepted['states']):
        out=root/'run'/f'rank-{rank}';out.mkdir(parents=True)
        torch.save(state,out/'checkpoint-0003.pt')
        put(out/'probabilities.json',ranks[rank]);put(out/'terminal.json',dict(exit_code=0))
        put(out/'result.json',dict(passed=True,rank=rank,updates=3,policy_id=accepted['policy_id'],
            reference_policy_id=accepted['reference_policy_id'],schedule_sha256=config['schedule_sha256'],
            checkpoint_id=accepted['checkpoint_id'],execution_dtypes={str(i):['torch.bfloat16']*2 for i in range(112)},
            exact_state=compare_state(state,copy.deepcopy(state),rank=rank,updates=3,continuous=True),parity=parity))
    path=Path(__file__).parents[1]/'tools/jev/verify_simple_continuous_restore.py'
    spec=importlib.util.spec_from_file_location('continuous_restore_audit',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,dict(training_root=str(root),training_observed=str(observed),pins={}),root,put


def test_fresh_process_state_and_probability_audit(tmp_path,monkeypatch):
    m,c,root,_=fixture(tmp_path,monkeypatch);r=m.run(c,tmp_path/'out')
    assert r['passed'] and r['exact_master_adam'] and r['requests']==128 and r['parity']['maximum_logprob_error']==0


@pytest.mark.parametrize('fault',['master','adam','cursor','dtype','probability','duplicate','partial',
    'reference','schedule','checkpoint','rank','claimed_parity','claimed_exact','live','terminal'])
def test_false_success_cannot_hide_recovery_drift(tmp_path,monkeypatch,fault):
    m,c,root,put=fixture(tmp_path,monkeypatch);out=root/'run/rank-1'
    if fault in ('master','adam','cursor'):
        p=out/'checkpoint-0003.pt';v=torch.load(p,weights_only=True)
        if fault=='master':
            k=v['names'][0];v['parameters'][k]=torch.nextafter(v['parameters'][k],torch.full_like(v['parameters'][k],float('inf')))
        elif fault=='adam':v['optimizer']['state'][0]['exp_avg']+=1e-7
        else:v['next_update']=2
        torch.save(v,p)
    else:
        if fault in ('probability','duplicate','partial'):p=out/'probabilities.json'
        elif fault=='live':p=Path(c['training_observed'])
        elif fault=='terminal':p=out/'terminal.json'
        else:p=out/'result.json'
        v=json.loads(p.read_text())
        if fault=='probability':v[0]['candidate_logprobs'][0][0]+=.01
        elif fault=='duplicate':v[0]=v[1]
        elif fault=='partial':v.pop()
        elif fault=='dtype':v['execution_dtypes']['0']=['torch.float32']*2
        elif fault=='reference':v['reference_policy_id']='moving'
        elif fault=='schedule':v['schedule_sha256']='other'
        elif fault=='checkpoint':v['checkpoint_id']='other'
        elif fault=='rank':v['rank']=2
        elif fault=='claimed_parity':v['parity']['maximum_logprob_error']=.1
        elif fault=='claimed_exact':v['exact_state']['exact']=False
        elif fault=='live':v['state']['Status']='running'
        else:v['exit_code']=1
        put(p,v)
    with pytest.raises(ValueError):m.run(c,tmp_path/'out')
