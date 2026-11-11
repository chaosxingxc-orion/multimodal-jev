"""Synthetic artifact binding tests; no actual initialization or NPU forward."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from copy import deepcopy
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/jev'))
from probe_omni_sampled_probability import RUN_NAMES, prerequisites
import probe_omni_sampled_probability as runner
from multimodal_jev.baseline import file_hash
from multimodal_jev.sfc_reward_batch import digest
from multimodal_jev.omni_probability_audit import validate_protocol
from test_omni_probability_audit import fixture


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def setup(tmp_path):
    args = SimpleNamespace(initialization=tmp_path/'init', preflight=tmp_path/'cpu',
        root=tmp_path/'sampling', model=tmp_path/'model', protocol=tmp_path/'protocol.json')
    pins = {'config.json': 'a'*64, 'model.safetensors': 'b'*64}
    assets = [dict(file=k, sha256=v) for k, v in pins.items()]
    policy = digest(dict(repository='Qwen/Qwen3-Omni-30B-A3B-Instruct',
        revision='26291f793822fb6be9555850f06dfe95f2d7e695', files=assets))
    write(args.preflight/'run/result.json', dict(model_files=pins))
    write(args.preflight/'acceptance.json', dict(result_sha256=file_hash(args.preflight/'run/result.json')))
    args.preflight_hash = file_hash(args.preflight/'acceptance.json')
    write(args.initialization/'run/original-parameters.json', {'fixture': True})
    rank_outputs = []
    for rank in (0, 1):
        row = dict(rank=rank)
        for name in ('result', 'parameters'):
            path = args.initialization/f'run/rank-{rank}'/(name+'.json')
            write(path, dict(rank=rank, synthetic=True))
            row[name+'_sha256'] = file_hash(path)
        rank_outputs.append(row)
    write(args.initialization/'acceptance.json', dict(
        schema='omni-fsdp-initialization-audit-v1', parameters=31719205488,
        parameter_tensors=1407, units=108, buffers=4, ranks=2, rank_outputs=rank_outputs,
        preflight_acceptance_sha256=args.preflight_hash,
        original_parameters_sha256=file_hash(args.initialization/'run/original-parameters.json')))
    args.initialization_hash = file_hash(args.initialization/'acceptance.json')
    write(args.root/'run/identity.json', dict(model_files=assets, policy_sha256=policy))
    records, _, protocol = fixture()
    for record in records:
        record['policy_sha256'] = policy
    protocol.update(policy_sha256=policy, records_sha256=digest(records), run_names=RUN_NAMES)
    write(args.protocol, protocol)
    args.protocol_hash = digest(protocol)
    return args, dict(audio_model=str(args.model)), records, protocol


def test_pre_forward_validation_needs_no_actor_results(tmp_path):
    args, base, records, protocol = setup(tmp_path)
    assert prerequisites(args, base, records) == protocol
    assert len(validate_protocol(records, protocol=protocol, protocol_sha256=args.protocol_hash)) == len(records)


@pytest.mark.parametrize('change', [
    'initialization_hash', 'preflight_hash', 'protocol_hash', 'rank_artifact',
    'cpu_result', 'original_parameters', 'model_path', 'model_files', 'duplicate_asset',
    'policy', 'missing_reference', 'invalid_limit', 'changed_records', 'missing_rank',
])
def test_rejects_changed_prerequisites_before_model_loading(tmp_path, change):
    args, base, records, protocol = setup(tmp_path)
    if change in ('initialization_hash', 'preflight_hash', 'protocol_hash'):
        setattr(args, change, '0'*64)
    elif change in ('rank_artifact', 'cpu_result', 'original_parameters'):
        paths = dict(rank_artifact=args.initialization/'run/rank-1/parameters.json',
            cpu_result=args.preflight/'run/result.json',
            original_parameters=args.initialization/'run/original-parameters.json')
        write(paths[change], {'changed': True})
    elif change == 'model_path':
        base['audio_model'] += '-other'
    elif change in ('model_files', 'duplicate_asset', 'policy'):
        path = args.root/'run/identity.json'
        data = json.loads(path.read_text())
        if change == 'model_files': data['model_files'][0]['sha256'] = 'd'*64
        if change == 'duplicate_asset': data['model_files'].append(data['model_files'][0])
        if change == 'policy': data['policy_sha256'] = 'e'*64
        write(path, data)
    elif change == 'missing_rank':
        path = args.initialization/'acceptance.json'
        data = json.loads(path.read_text()); data['rank_outputs'].pop()
        write(path, data); args.initialization_hash = file_hash(path)
    elif change == 'changed_records':
        records[0]['log_probs'][0] -= .5
    else:
        if change == 'missing_reference': protocol['run_names'] = ['micro1', 'micro2']
        if change == 'invalid_limit': protocol['token_ratio_lower'] = 0
        write(args.protocol, protocol); args.protocol_hash = digest(protocol)
    with pytest.raises(ValueError):
        prerequisites(args, base, records)


@pytest.mark.parametrize('mode', ['pass', 'numerical_failure', 'initialization_failure', 'rank_disagreement'])
def test_same_verified_model_complete_merge_and_failures(tmp_path, monkeypatch, mode):
    """Exercise orchestration with synthetic workers, never claim real forward."""
    import torch

    args, base, records, protocol = setup(tmp_path)
    records = [dict(deepcopy(records[i % 3]), example_id=f'example-{i}') for i in range(8)]
    protocol['records_sha256'] = digest(records)
    write(args.protocol, protocol); args.protocol_hash = digest(protocol)
    args.output = tmp_path/'output'
    args.acceptance = tmp_path/'sampling-acceptance.json'
    args.acceptance_hash = 'c'*64
    out = args.output/'rank-0'; out.mkdir(parents=True)
    batch = object(); model = SimpleNamespace(training=False, parameters=lambda: [])
    calls = []

    def prepare(pre):
        assert pre.root == args.root
        write(pre.output/'result.json', dict(tensors={'synthetic': True},
            prepared_inputs_sha256='a'*64, processor_parity_sha256='b'*64,
            sampling_identity_sha256='c'*64))
        return SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)), records, batch, base

    def initialize(init_args, rank, local_rank, init_out, *, after_verification):
        calls.append('initialization')
        assert rank == local_rank == 0 and init_args.model == args.model
        if mode == 'initialization_failure': raise ValueError('synthetic initialization failure')
        calls.append('verified')
        after_verification(model)

    class Actor:
        def __init__(self, config, wrapped):
            assert wrapped is model and calls[-1] == 'verified'
            assert not config['use_dynamic_bsz'] and not config['use_remove_padding']

    actor_module = ModuleType('verl.workers.actor.dp_actor')
    actor_module.DataParallelPPOActor = Actor
    monkeypatch.setitem(sys.modules, actor_module.__name__, actor_module)
    omega = ModuleType('omegaconf'); omega.OmegaConf = SimpleNamespace(create=lambda x: x)
    monkeypatch.setitem(sys.modules, 'omegaconf', omega)
    monkeypatch.setattr(runner, 'PINNED_ACTOR_SOURCE', file_hash(__file__))
    monkeypatch.setattr(runner, 'prepare_inputs', prepare)
    monkeypatch.setattr(runner, 'initialize', initialize)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda: None,
        max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200), raising=False)

    def gather(output, value):
        output[:] = [value, 'different' if mode == 'rank_disagreement' else value]

    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    monkeypatch.setattr(torch.distributed, 'barrier', lambda: None)

    def recompute(actor, actual_batch, actual_records, *, name, rank, plan, plan_sha256, pad_token_id):
        assert isinstance(actor, Actor) and actual_batch is batch and actual_records is records
        assert calls[:2] == ['initialization', 'verified'] and rank == 0
        calls.append(name)
        shards = []
        for peer in (0, 1):
            responses = []
            for i in plan['rank_indices'][peer]:
                record = records[i]
                responses.append({k: deepcopy(record[k]) for k in
                    ('example_id', 'decision_id', 'replicate', 'token_ids', 'log_probs')})
                responses[-1]['record_sha256'] = digest(record)
            if mode == 'numerical_failure' and peer == 1:
                responses[-1]['log_probs'][-1] -= 2
            def envelope(run_name):
                return dict(schema='omni-probability-shard-output-v1', rank=peer,
                    plan_sha256=plan_sha256, records_sha256=digest(records),
                    run=dict(name=run_name, responses=deepcopy(responses)))
            shards.append(dict(native=envelope(name), reference=envelope(name+'-same-logits-fp32')))
        write(args.output/'rank-1'/(name+'.json'), shards[1])
        return shards[0]

    monkeypatch.setattr(runner, 'recompute_observed_shard', recompute)
    if mode == 'pass':
        runner.run(args, 0, 0, out)
    else:
        with pytest.raises(ValueError): runner.run(args, 0, 0, out)
    if mode in ('pass', 'numerical_failure'):
        report = json.loads((args.output/'probability-audit.json').read_text())
        assert report['numerical_gate_passed'] == (mode == 'pass')
        assert report['responses'] == 8 and len(report['runs']) == 4
        assert calls == ['initialization', 'verified', 'micro1', 'micro2']
    else:
        assert not (args.output/'probability-audit.json').exists()
        assert not any(c.startswith('micro') for c in calls)
