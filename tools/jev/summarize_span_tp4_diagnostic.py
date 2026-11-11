#!/usr/bin/env python3
"""Recompute experimental TP4 candidate parity; never issue training READY."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_parity import compare_span_policies
from multimodal_jev.span_runtime_topology import TP4_HF_ORDER, TP4_COMPLETE_K, TP4_FULL_WIDTH, runtime_topology


def summarize(root):
    config = json.loads((root / 'config.json').read_text())
    assert runtime_topology(config)['diagnostic_only']
    launch = json.loads((root / 'launch.json').read_text())
    assert launch['config_sha256'] == file_hash(root / 'config.json')
    assert launch['source_manifest_sha256'] == file_hash(root / 'source-manifest.json')
    assert launch['physical_devices'] == [4, 5, 6, 7]
    for path, digest in config['pins'].items():
        assert file_hash(path) == digest, path
    terminal = json.loads((root / 'run/terminal.json').read_text())
    assert terminal['exit_code'] == 0
    result = json.loads((root / 'run/result.json').read_text())
    assert result['runtime_profile'] == config['runtime_profile']
    assert result['diagnostic_only'] is True and result['tensor_parallel_size'] == 4
    assert result['base_id'] == checkpoint_identity(config['model'])[0]
    assert result['adapter_id'] == checkpoint_identity(config['adapter'])[0]
    reference = json.loads(Path(config['hf_results']).read_text())
    assert reference['base_id'] == result['base_id'] and reference['adapter_id'] == result['adapter_id']
    expected = {r['id']: r for r in reference['records']}
    actual = {r['id']: r for r in result['records']}
    assert len(expected) == len(actual) == len(result['records']) == 48
    assert expected.keys() == actual.keys()
    gates = []
    for key, row in actual.items():
        hf = expected[key]
        assert row['request_id'] == hf['request_id']
        assert row['request_identity'] == hf['request_identity']
        # Independent HF suite stores one distribution per question.
        probs = hf['candidate_logprobs']
        if isinstance(probs[0][0], list):
            probs = [p[0] for p in probs]
        assert probs == row['hf_candidate_logprobs']
        gate = compare_span_policies(probs, row['candidate_logprobs'])
        assert gate == row['parity']
        gates.append(gate)
    assert result['passed'] == all(g['passed'] for g in gates)
    workers = json.loads((root / 'run/worker.json').read_text())
    profile = json.loads((root / 'run/runtime-profile.json').read_text())
    assert profile['profile'] == config['runtime_profile'] and profile['diagnostic_only'] is True
    receipts = profile['operations']['align_jev_internal_format']
    assert len(receipts) == 4 and all(r['after'] == 'disable' for r in receipts)
    if config['runtime_profile'] in (TP4_HF_ORDER, TP4_COMPLETE_K, TP4_FULL_WIDTH):
        aligned = profile['operations']['align_jev_tp4_hf_order']
        assert len(aligned) == 4
        for receipt in aligned:
            assert receipt['tensor_parallel_size'] == 4 and receipt['diagnostic_only'] is True
            for name in ('qkv', 'mlp', 'attention'):
                assert receipt[name]['tensor_parallel_size'] == 4 and receipt[name]['layers'] == 28
    projection_cost=None
    if config['runtime_profile'] in (TP4_COMPLETE_K, TP4_FULL_WIDTH):
        receipts=profile['operations']['align_jev_tp4_complete_k']
        assert sorted(r['rank'] for r in receipts)==list(range(4))
        for receipt in receipts:
            assert len(receipt['modules'])==56
            for module in receipt['modules']:
                old,new=module['old_shape'],module['new_shape']
                assert new==[old[0]//4,old[1]*4]
        for worker in workers:
            assert len(worker['complete_k_projections'])==56
            assert all(m['calls']>=48 for m in worker['complete_k_projections'].values())
        projection_cost=dict(
            modules_per_rank=56,
            local_weight_bytes_per_rank=[sum(m['local_weight_bytes'] for m in r['modules']) for r in receipts],
            maximum_single_full_weight_bytes=max(m['transient_full_weight_bytes'] for r in receipts for m in r['modules']),
            calls_per_module=sorted({m['calls'] for w in workers for m in w['complete_k_projections'].values()}),
            maximum_single_gathered_input_bytes=max(m['maximum_gathered_input_bytes'] for w in workers for m in w['complete_k_projections'].values()),
            maximum_single_gathered_output_bytes=max(m['maximum_gathered_output_bytes'] for w in workers for m in w['complete_k_projections'].values()),
            scope='Tensor payload/storage accounting only; not wire traffic, measured peak HBM or communication latency')
    if config['runtime_profile']==TP4_FULL_WIDTH:
        receipts=profile['operations']['align_jev_tp4_full_width']
        assert sorted(r['rank'] for r in receipts)==list(range(4))
        assert all(r['layers']==28 and r['diagnostic_only'] for r in receipts)
        for worker in workers:
            assert len(worker['full_width_projections'])==56
            assert all(c>=48 for c in worker['full_width_projections'].values())
        projection_cost['maximum_single_padded_weight_bytes']=max(r['maximum_single_padded_weight_bytes'] for r in receipts)
    assert len(workers) == 4
    for worker in workers:
        assert worker['model_class'] == 'Qwen3ASRForSpanDecisionsTP4Diagnostic'
        assert worker['lora_dtype'] == worker['head_projection_dtype'] == 'torch.bfloat16'
        assert worker['registered_adapter_ids'] == [1]
        assert len(worker['wrapper_modules']) == 28
        assert all(v == ['torch.bfloat16'] for v in worker['actual_lora_buffer_dtypes'].values())
    return dict(root=str(root), diagnostic_only=True, training_admitted=False,
                projection_cost=projection_cost,
                runtime_profile=config['runtime_profile'], adapter_id=result['adapter_id'],
                records=len(gates), passed_records=sum(g['passed'] for g in gates),
                maximum_logprob_error=max(g['maximum_logprob_error'] for g in gates),
                joint_ratio_range=[min(g['joint_ratio_min'] for g in gates),
                                   max(g['joint_ratio_max'] for g in gates)],
                argmax_changed_spans=sum(len(g['argmax_changed_spans']) for g in gates),
                files={str(p): file_hash(p) for p in [root/'run/result.json', root/'run/worker.json',
                       root/'run/terminal.json', root/'run/runtime-profile.json',
                       root/'config.json', root/'source-manifest.json']},
                scope='Independent TP4 diagnostic; no RL admission or task improvement claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
