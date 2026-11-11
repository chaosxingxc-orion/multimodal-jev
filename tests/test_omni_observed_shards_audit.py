"""Saved synthetic normalization evidence; never actual NPU execution."""
from copy import deepcopy

import pytest
import torch

from multimodal_jev.omni_logprob_observer import (
    PINNED_SOURCE, PINNED_ACTOR_SOURCE, bind_observations,
)
from multimodal_jev.omni_probability_shards import (
    _local_batch, audit_observed_shards, shard_plan,
)
from multimodal_jev.sfc_reward_batch import digest
from test_omni_probability_shards import setup


def fixture(size, real=False):
    records, data = setup(real)
    plan = shard_plan(records, world_size=2, micro_batch_size=size)
    outputs = []
    for rank in (0, 1):
        local, rows = _local_batch(data, records, rank=rank, plan=plan, plan_sha256=digest(plan))
        tokens = local.batch['responses']; native = -tokens.double()/8-.125
        report = dict(source_sha256=PINNED_SOURCE, actor_source_sha256=PINNED_ACTOR_SOURCE,
            flash_available=False, npu_available=True, calls=[])
        for index, start in enumerate(range(0, len(rows), size)):
            labels = tokens[start:start+size]; values = native[start:start+size]
            report['calls'].append(dict(call=index, logits_shape=[*labels.shape, 128],
                positions=labels.numel(), labels_dtype='torch.int64',
                logits_device=f'npu:{rank}', output_device=f'npu:{rank}',
                reference_dtype='torch.float32', reference_device='cpu',
                label_ids=labels.tolist(), native_log_probs=values.tolist(),
                reference_log_probs=(values.float()-.01).tolist()))
        observed = bind_observations(local, native, rows, report, name='micro', micro_batch_size=size)
        def envelope(run):
            return dict(schema='omni-probability-shard-output-v1', rank=rank,
                plan_sha256=digest(plan), records_sha256=digest(records), run=run)
        outputs.append(dict(native=envelope(deepcopy(observed['native'])),
            reference=envelope(deepcopy(observed['reference'])), observation=observed))
    return data, records, outputs, dict(name='micro', plan=plan, plan_sha256=digest(plan))


@pytest.mark.parametrize('size', [1, 2])
@pytest.mark.parametrize('real', [False, True])
def test_rechecks_all_calls_against_original_inputs_and_both_exports(size, real):
    if real:
        pytest.importorskip('verl')
    data, records, outputs, kwargs = fixture(size, real)
    before = deepcopy(outputs)
    result = audit_observed_shards(data[[2, 0, 3, 1]], records, list(reversed(outputs)), **kwargs)
    assert outputs == before
    assert result['observed_ranks'] == 2 and result['normalization_calls'] == 4//size
    assert len(result['native']['responses']) == len(records)


@pytest.mark.parametrize('change', [
    'native_call', 'reference_call', 'labels', 'mask', 'identity', 'record_binding',
    'missing_call', 'missing_rank', 'source', 'cpu_device', 'native_export',
    'reference_export', 'reference_rank', 'observation_native', 'observation_reference', 'wrong_device_rank',
])
def test_rejects_observation_corruption_even_when_export_coverage_passes(change):
    data, records, outputs, kwargs = fixture(1)
    observation = outputs[0]['observation']; call = observation['observer']['calls'][0]
    binding = observation['microbatch_bindings'][0]
    if change == 'native_call': call['native_log_probs'][0][0] -= .5
    if change == 'reference_call': call['reference_log_probs'][0][0] -= .5
    if change == 'labels': call['label_ids'][0][0] += 1
    if change == 'mask': binding['response_mask'][0][0] = False
    if change == 'identity': binding['record_identities'][0][0] = 'other'
    if change == 'record_binding': binding['record_sha256'][0] = 'f'*64
    if change == 'missing_call': observation['observer']['calls'].pop()
    if change == 'missing_rank': outputs.pop()
    if change == 'source': observation['observer']['source_sha256'] = 'f'*64
    if change == 'cpu_device': call['logits_device'] = call['output_device'] = 'cpu'
    if change == 'wrong_device_rank': call['logits_device'] = call['output_device'] = 'npu:1'
    if change == 'native_export': outputs[0]['native']['run']['responses'][0]['log_probs'][0] -= .5
    if change == 'reference_export': outputs[0]['reference']['run']['responses'][0]['log_probs'][0] -= .5
    if change == 'reference_rank':
        outputs[0]['reference'], outputs[1]['reference'] = outputs[1]['reference'], outputs[0]['reference']
    if change == 'observation_native': observation['native']['responses'][0]['log_probs'][0] -= .5
    if change == 'observation_reference': observation['reference']['responses'][0]['log_probs'][0] -= .5
    with pytest.raises(ValueError):
        audit_observed_shards(data, records, outputs, **kwargs)


@pytest.mark.parametrize('change', [None, 'terminal', 'binding', 'merged', 'numeric', 'observation'])
def test_offline_audit_rebuilds_inputs_and_reproduces_complete_evidence(tmp_path, monkeypatch, change):
    import json
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/jev'))
    import audit_omni_sampled_probability as audit
    from multimodal_jev.omni_probability_audit import compare

    def write(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    data, records, _, _ = fixture(1)
    args = SimpleNamespace(output=tmp_path/'audit', root=tmp_path/'run', sampling=tmp_path/'sampling',
        acceptance=tmp_path/'accepted.json', acceptance_hash='a'*64, protocol=tmp_path/'protocol.json')
    args.output.mkdir()
    protocol = dict(schema='omni-probability-comparison-v1', records_sha256=digest(records),
        policy_sha256=records[0]['policy_sha256'], instruction_sha256=records[0]['instruction_sha256'],
        run_names=['micro1', 'micro1-same-logits-fp32', 'micro2', 'micro2-same-logits-fp32'],
        max_abs_logprob_error=.2, max_between_run_logprob_error=.02,
        token_ratio_lower=.8, token_ratio_upper=1.2)
    args.protocol_hash = digest(protocol); write(args.protocol, protocol)
    inputs = dict(tensors={'fixture': True}, prepared_inputs_sha256='b'*64,
        processor_parity_sha256='c'*64, sampling_identity_sha256='d'*64)
    def prepare(pre):
        assert pre.root == args.sampling
        write(pre.output/'result.json', inputs)
        return None, records, data, None
    monkeypatch.setattr(audit, 'prepare_inputs', prepare)
    plans = [shard_plan(records, world_size=2, micro_batch_size=size) for size in (1, 2)]
    binding = dict(inputs, protocol=protocol, protocol_sha256=args.protocol_hash, plans=plans,
        sampling_acceptance_sha256=args.acceptance_hash)
    for rank in (0, 1):
        root = args.root/f'rank-{rank}'
        write(root/'terminal.json', dict(exit_code=0))
        write(root/'registered-inputs.json', dict(binding=binding))
        write(root/'result.json', dict(status='probabilities_captured', rank=rank,
            binding_sha256=digest(binding), weight_updates=0, targets_read=False, forward_calls=3))
    runs = []
    for size in (1, 2):
        _, _, outputs, kwargs = fixture(size)
        name = f'micro{size}'; kwargs['name'] = name
        for rank, output in enumerate(outputs):
            for key, suffix in [('native', ''), ('reference', '-same-logits-fp32')]:
                output[key]['run']['name'] = name+suffix
                output['observation'][key]['name'] = name+suffix
            write(args.root/f'rank-{rank}'/(name+'.json'), output)
        checked = audit_observed_shards(data, records, outputs, **kwargs)
        runs.extend([checked['native'], checked['reference']])
    write(args.root/'merged-runs.json', runs)
    report = compare(records, runs, protocol=protocol, protocol_sha256=args.protocol_hash)
    assert report['numerical_gate_passed']
    write(args.root/'probability-audit.json', report)
    if change == 'terminal': write(args.root/'rank-1/terminal.json', dict(exit_code=1))
    if change == 'binding':
        binding['tensors'] = {'changed': True}
        write(args.root/'rank-1/registered-inputs.json', dict(binding=binding))
    if change == 'merged': write(args.root/'merged-runs.json', runs[:-1])
    if change == 'numeric':
        report['tokens'] += 1; write(args.root/'probability-audit.json', report)
    if change == 'observation':
        path = args.root/'rank-1/micro2.json'; output = json.loads(path.read_text())
        output['observation']['microbatch_bindings'][0]['record_sha256'][0] = 'f'*64
        write(path, output)
    if change:
        with pytest.raises(ValueError): audit.run(args)
        assert not (args.output/'result.json').exists()
    else:
        audit.run(args)
        assert json.loads((args.output/'result.json').read_text())['responses'] == len(records)
