"""A forged passed/READY artifact must not hide state or reference drift."""
import importlib.util
import itertools
from pathlib import Path
import pytest
import torch
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_runtime_profile import PROFILE

spec=importlib.util.spec_from_file_location('chain_verifier',Path(__file__).parents[1]/'tools/jev/verify_span_training_chain.py')
verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)


def archive(root):
    base=root/'base';base.mkdir();(base/'weights.safetensors').write_bytes(b'base')
    base_id=checkpoint_identity(base)[0]
    def checkpoint(path,number):
        adapter=path/'adapter';adapter.mkdir(parents=True)
        (adapter/'weights.safetensors').write_bytes(str(number).encode())
        aid,files=checkpoint_identity(adapter)
        cursor=dict(schema='cursor',seed=17,schedule_sha256='fixed',epoch_batches=20,
            manifest_sha256='data',question_order=['date','time','timeofday'],batch_index=number-1,next_batch_index=number)
        states={i:dict(step=torch.tensor(float(number)),exp_avg=torch.zeros(1),exp_avg_sq=torch.zeros(1)) for i in range(112)}
        torch.save(dict(completed_updates=number,training_state=cursor,optimizer=dict(optimizer=dict(state=states))),path/'optimizer.pt')
        receipt=dict(schema='jev-span-checkpoint-v1',base_id=base_id,adapter_id=aid,adapter_files=files,
            completed_updates=number,training_state=cursor,optimizer_sha256=file_hash(path/'optimizer.pt'))
        atomic_json(path/'CHECKPOINT.json',receipt)
        return receipt
    def policy(path,adapter):
        path.parent.mkdir(parents=True,exist_ok=True)
        labels=list(itertools.product((False,True),repeat=3))
        records=[dict(id=str(i),request_id=str(i),request_identity={'input':i},labels=labels[i%8],supervised=[True]*3,
            hf_candidate_logprobs=[[-0.69314718056]*2]*3,candidate_logprobs=[[-0.69314718056]*2]*3) for i in range(48)]
        result=dict(runtime_profile=PROFILE,passed=True,base_id=base_id,adapter_id=checkpoint_identity(adapter)[0],records=records)
        atomic_json(path,result);return result
    prior=root/'prior';old=checkpoint(prior,2)
    initial=root/'initial.json';policy(initial,prior/'adapter')
    step=dict(model_path=str(base),initial_adapter=str(prior/'adapter'),initial_optimizer_checkpoint=str(prior),
        runtime_profile=PROFILE,runtime_acceptance=str(initial),pins={})
    step['pins']={str(p):file_hash(p) for p in [*prior.rglob('*'),initial] if p.is_file()}
    atomic_json(root/'config.json',dict(num_updates=1,step=step))
    atomic_json(root/'source-manifest.json',{})
    atomic_json(root/'launch.json',dict(physical_devices=[4],config_sha256=file_hash(root/'config.json'),source_manifest_sha256=file_hash(root/'source-manifest.json')))
    run=root/'run';training=run/'update-0003';suite=run/'acceptance-0003'
    receipt=checkpoint(training/'policy-1',3)
    atomic_json(training.with_suffix('.config.json'),step)
    atomic_json(training/'selection.json',receipt['training_state'])
    atomic_json(training/'optimizer-resume.json',dict(previous_updates=2,optimizer_sha256=old['optimizer_sha256'],optimizer=dict(optimizer_steps=[2])))
    atomic_json(training/'export.json',dict(optimizer=dict(optimizer_steps=[3])))
    atomic_json(training/'actor-reload.json',dict(maximum_logprob_error=0))
    atomic_json(training/'update.json',dict(gradient_norm=1.))
    result=policy(suite/'vllm/result.json',training/'policy-1/adapter')
    (suite/'hf').mkdir()
    atomic_json(suite/'hf/result.json',result)
    for p in (run,training,suite):
        atomic_json(p/'terminal.json',dict(exit_code=0));atomic_json(p/'READY.json',dict(status='passed',completed_updates=3))
    cp=training/'policy-1';acceptance=suite/'vllm/result.json'
    version=dict(completed_updates=3,checkpoint=str(cp),adapter_id=receipt['adapter_id'],optimizer_sha256=receipt['optimizer_sha256'],
        checkpoint_receipt_sha256=file_hash(cp/'CHECKPOINT.json'),runtime_acceptance=str(acceptance),acceptance_sha256=file_hash(acceptance),next_batch_index=3)
    atomic_json(run/'accepted-versions.json',[version]);atomic_json(run/'last-accepted.json',version)
    return run,training,suite


def test_recomputes_valid_archive(tmp_path):
    archive(tmp_path)
    result=verifier.verify(tmp_path)
    assert result['completed_updates']==3 and result['versions'][0]['recordings']==48


@pytest.mark.parametrize('fault',['hf','adam','terminal','lineage'])
def test_forged_success_does_not_hide_drift(tmp_path,fault):
    run,training,suite=archive(tmp_path)
    if fault=='hf':
        p=suite/'hf/result.json';d=verifier.read(p);d['records'][0]['candidate_logprobs'][0]=[-0.5,-0.9];atomic_json(p,d)
    elif fault=='adam':
        p=training/'policy-1/optimizer.pt';d=torch.load(p,weights_only=True);d['optimizer']['optimizer']['state'][0]['step']=torch.tensor(2.);torch.save(d,p)
        # Even updating the checkpoint hash must not hide a stale moment step.
        p=training/'policy-1/CHECKPOINT.json';d=verifier.read(p);d['optimizer_sha256']=file_hash(p.parent/'optimizer.pt');atomic_json(p,d)
    elif fault=='terminal':(suite/'terminal.json').unlink()
    else:
        p=training.with_suffix('.config.json');d=verifier.read(p);d['initial_optimizer_checkpoint']='different';atomic_json(p,d)
    with pytest.raises((ValueError,FileNotFoundError)):verifier.verify(tmp_path)
