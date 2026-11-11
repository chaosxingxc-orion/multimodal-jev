"""Controller fault injection; fake children do not prove NPU numerics."""
import json
from pathlib import Path
import pytest
from multimodal_jev import span_training_chain as chain
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.policy_identity import checkpoint_identity


def checkpoint(root, updates, cursor, base):
    (root/'adapter').mkdir(parents=True)
    (root/'adapter/adapter_model.safetensors').write_bytes(str(updates).encode())
    atomic_json(root/'adapter/adapter_config.json', {})
    (root/'optimizer.pt').write_bytes(b'fake-optimizer')
    adapter, files = checkpoint_identity(root/'adapter')
    atomic_json(root/'CHECKPOINT.json', dict(schema='jev-span-checkpoint-v1', base_id=base,
        adapter_id=adapter, adapter_files=files, completed_updates=updates,
        optimizer_sha256=file_hash(root/'optimizer.pt'), training_state=cursor))


def setup(tmp_path, monkeypatch, fault=None):
    base = tmp_path/'base';base.mkdir();(base/'model.safetensors').write_bytes(b'base')
    base_id = checkpoint_identity(base)[0]
    cursor = dict(schema='jev-span-balanced-cursor-v1', seed=17, manifest_sha256='data',
        question_order=['date','time','timeofday'], schedule_sha256='schedule', epoch_batches=20,
        batch_index=1, next_batch_index=2, recording_ids=['old-a','old-b'])
    initial = tmp_path/'initial';checkpoint(initial, 2, cursor, base_id)
    identity = tmp_path/'identity.json';atomic_json(identity, {})
    acceptance = tmp_path/'initial-acceptance.json';atomic_json(acceptance, {'passed':True})
    pins = {str(p):file_hash(p) for p in [*initial.rglob('*'),identity,acceptance] if p.is_file()}
    step = dict(initial_adapter=str(initial/'adapter'), initial_optimizer_checkpoint=str(initial),
        model_path=str(base), batch_index=2, seed=17, manifest_sha256='data', question_order=cursor['question_order'],
        runtime_profile='fixture', runtime_acceptance=str(acceptance), pins=pins)
    config = dict(num_updates=2, suite_identity=str(identity), step=step)
    calls=[];admissions=[]
    def admit(config):
        chain.check_pins(config['pins'])
        admissions.append((config['initial_adapter'],config['runtime_acceptance']))
    monkeypatch.setattr(chain,'verify_suite_admission',admit)
    def child(argv, stdout, stderr, check):
        assert check
        config=chain.read(argv[argv.index('--config')+1]);dest=Path(argv[argv.index('--output')+1]);dest.mkdir()
        script=Path(argv[1]).name;calls.append(script)
        if script=='train_span_rl_step.py':
            old=chain.read(Path(config['initial_optimizer_checkpoint'])/'CHECKPOINT.json')
            index=config['batch_index'];assert index==old['training_state']['next_batch_index']
            assert old['completed_updates']==index
            if index==3:
                assert 'acceptance-0003' in config['runtime_acceptance']
                assert 'update-0003' in config['initial_adapter']
            state={**cursor,'batch_index':index,'next_batch_index':index+1,'recording_ids':[f'{index}-a',f'{index}-b']}
            if fault=='cursor':state['schedule_sha256']='changed'
            checkpoint(dest/'policy-1',old['completed_updates']+1,state,base_id)
            atomic_json(dest/'selection.json',state)
            ready=dict(status='passed')
        else:
            (dest/'vllm').mkdir()
            adapter=checkpoint_identity(config['adapter'])[0]
            atomic_json(dest/'vllm/result.json',dict(passed=True,adapter_id=adapter))
            ready=dict(status='passed',base_id=base_id,adapter_id=adapter,recordings=48,
                result_sha256=file_hash(dest/'vllm/result.json'))
            if fault=='suite_identity':ready['adapter_id']='stale'
            if fault=='second_suite' and '0004' in dest.name:
                raise RuntimeError('injected parity failure')
        atomic_json(dest/'READY.json',ready)
        if fault!='missing_terminal':atomic_json(dest/'terminal.json',dict(exit_code=0))
    monkeypatch.setattr(chain.subprocess,'run',child)
    output=tmp_path/'run';output.mkdir()
    return config,output,calls,admissions


def test_two_steps_chain_exact_versions_and_full_suites(tmp_path,monkeypatch):
    config,output,calls,admissions=setup(tmp_path,monkeypatch)
    chain.run_chain(config,output,tmp_path)
    assert calls==['train_span_rl_step.py','check_span_policy_version.py']*2
    assert len(admissions)==3
    result=chain.read(output/'READY.json')
    assert result['initial_completed_updates']==2 and result['completed_updates']==4
    assert result['latest']['next_batch_index']==4
    assert len(chain.read(output/'accepted-versions.json'))==2


@pytest.mark.parametrize('fault',['cursor','suite_identity','missing_terminal','second_suite'])
def test_failure_never_advances_unaccepted_policy(tmp_path,monkeypatch,fault):
    config,output,calls,_=setup(tmp_path,monkeypatch,fault)
    with pytest.raises((ValueError,FileNotFoundError,RuntimeError)):
        chain.run_chain(config,output,tmp_path)
    assert not (output/'READY.json').exists()
    if fault=='second_suite':
        assert chain.read(output/'last-accepted.json')['completed_updates']==3
        assert len(chain.read(output/'accepted-versions.json'))==1
    else:
        assert not (output/'last-accepted.json').exists()
        assert calls.count('train_span_rl_step.py')==1


@pytest.mark.parametrize('fault',['unpin','tamper','epoch_overflow','repeat','bool_updates'])
def test_preflight_rejects_before_starting_children(tmp_path,monkeypatch,fault):
    config,output,calls,_=setup(tmp_path,monkeypatch)
    checkpoint_path=Path(config['step']['initial_optimizer_checkpoint'])/'optimizer.pt'
    if fault=='unpin':del config['step']['pins'][str(checkpoint_path)]
    if fault=='tamper':checkpoint_path.write_bytes(b'changed')
    if fault=='epoch_overflow':config['num_updates']=19
    if fault=='repeat':config['step']['batch_index']=1
    if fault=='bool_updates':config['num_updates']=True
    with pytest.raises(ValueError):chain.run_chain(config,output,tmp_path)
    assert not calls
