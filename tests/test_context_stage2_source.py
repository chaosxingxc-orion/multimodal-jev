import json
from pathlib import Path

import pytest

from tools.jev import context_stage2_source as module
from tools.jev.context_rl_evaluation_plan import sha


@pytest.fixture
def sources(tmp_path, monkeypatch):
    experiment = json.loads(Path('configs/releases/context-prob-v1/experiment.json').read_text())
    roots = [tmp_path / 'training', tmp_path / 'selection']
    proofs = [dict(state=dict(Status='exited', ExitCode=0, OOMKilled=False), files={}) for _ in roots]
    def put(i, name, value):
        path = roots[i] / name;path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value));proofs[i]['files'][name] = sha(path)
        return path
    contract = dict(base_id='base', token={'registration_status':'admitted'}, request_version='jev-context-exclusive_event-v1')
    put(0, 'config.json', dict(stage='instruction', engineering_prefix=False, training_dispatch_allowed=True,
        stop_update=634, release_sha256=experiment['release_sha256'], source_contract=contract, base_id='base'))
    put(0, 'run/result.json', dict(passed=True, stop_update=634, schedule_updates=634,
        learning_population=40513, engineering_prefix=False))
    for i, pid, directory in ((0, 'source-id', 'run/policy-0634'), (1, 'export-id', 'run/policy')):
        asset = put(i, directory + '/execution.pt', {'fixture':'same tensors'})
        put(i, directory + '/MANIFEST.json', dict(policy_id=pid, contract=contract, files={'execution.pt':sha(asset)}))
    shards = {}
    for rank in range(4):
        name = f'rank-{rank}/checkpoint-0634.pt';shards[name] = sha(put(0, 'run/' + name, {'fixture':rank}))
    execution = put(0, 'run/execution-0634.pt', {'fixture':'execution'})
    put(0, 'run/checkpoint-0634-committed.json', dict(updates=634, contract={'stage':'instruction'},
        joint_policy_directory='policy-0634', joint_policy={'policy_id':'source-id'},
        joint_policy_manifest_sha256=proofs[0]['files']['run/policy-0634/MANIFEST.json'],
        rank_shards=shards, execution_file='execution-0634.pt', execution_sha256=sha(execution)))
    proofs[0].update(admitted_joint_policy_checkpoints=10, checkpoint_steps=[*range(64,634,64),634])
    put(1, 'config.json', dict(checkpoint_step=634, source_policy_id='source-id', source_contract=contract,
        release_sha256=experiment['release_sha256'], audio=[dict(id=str(i)) for i in range(256)]))
    put(1, 'run/joint.json', dict(policy_id='export-id', contract=contract))
    hf = [dict(id=str(i), inputs={'fixture':i}, positions=[list(range(60))], prob=[.5]*60) for i in range(256)]
    vl = [dict(id=x['id'], request_identity=dict(inputs=x['inputs'], positions=x['positions'][0], policy_id='export-id'), prob=x['prob']) for x in hf]
    put(1, 'run/hf.json', hf);put(1, 'run/vllm.json', {'records':vl})
    put(1, 'selection/predictions.json', {'records':[dict(id=str(i), policy_id='export-id') for i in range(9226)]})
    put(1, 'selection/input-receipts.json', [dict(id=str(i)) for i in range(9226)])
    proofs[1].update(step=634, source_policy_id='source-id', inference_policy_id='export-id',
        source_tensors_exact=True, parity_queries=256, parity_max_error=0,
        result=dict(role='selection', policy_id='export-id', release_sha256=experiment['release_sha256'],
                    predictions_sha256=proofs[1]['files']['selection/predictions.json']))
    # Binding tests only; no claim of model tensor authentication.
    monkeypatch.setattr(module, 'authenticate', lambda *args: None)
    return dict(experiment=experiment, training_root=roots[0], training_proof=proofs[0],
                selection_root=roots[1], selection_proof=proofs[1])


def test_fixed_endpoint_shared_by_algorithms(sources):
    result = module.bind(**sources)
    assert result['source_policy_id'] == result['reference_policy_id'] == 'source-id'
    assert result['evaluated_policy_id'] == 'export-id'
    assert result['algorithms'] == ['grpo_v1', 'rloo_v1']
    assert result['source_step'] == 634 and result['training_dispatch_allowed'] is False


@pytest.mark.parametrize('fault', ['live', 'oom', 'partial', 'wrong_step', 'wrong_source',
    'missing_shard', 'changed_asset', 'wrong_stage', 'missing_eval', 'bad_parity', 'bad_metrics'])
def test_reject_incomplete_or_mixed_stage1(sources, fault):
    if fault == 'live':sources['training_proof']['state']['Status'] = 'running'
    elif fault == 'oom':sources['selection_proof']['state']['OOMKilled'] = True
    elif fault == 'missing_shard':del sources['training_proof']['files']['run/rank-3/checkpoint-0634.pt']
    elif fault == 'changed_asset':(sources['training_root'] / 'run/policy-0634/execution.pt').write_text('changed')
    elif fault == 'missing_eval':del sources['selection_proof']['files']['selection/predictions.json']
    elif fault == 'bad_metrics':sources['selection_proof']['result']['predictions_sha256'] = 'wrong'
    else:
        role, name = ('training', 'config.json') if fault in ('partial','wrong_stage') else ('selection', 'config.json')
        if fault == 'bad_parity':name = 'run/hf.json'
        path = sources[role+'_root'] / name;data = json.loads(path.read_text())
        if fault == 'partial':data['engineering_prefix'] = True
        elif fault == 'wrong_stage':data['stage'] = 'task'
        elif fault == 'wrong_step':data['checkpoint_step'] = 128
        elif fault == 'wrong_source':data['source_policy_id'] = 'old-step2'
        elif fault == 'bad_parity':data[0]['prob'][0] = .9
        path.write_text(json.dumps(data));sources[role+'_proof']['files'][name] = sha(path)
    with pytest.raises((ValueError, FileNotFoundError)):module.bind(**sources)
