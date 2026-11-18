"""Candidate-only authenticated continuation and same-checkpoint batch checks."""
import json
from pathlib import Path

from .baseline import file_hash, atomic_json
from .span_batch_training_config import batch_layout


def validate_lineage(previous, current, state):
    batch_layout(previous); batch_layout(current)
    fields = ('model_path', 'base_id', 'release', 'release_sha256', 'schedule_sha256',
              'global_batch', 'microbatch_per_rank', 'batch_order', 'candidate_profile',
              'lora_rank', 'lora_alpha', 'lora_execution_dtype', 'lr', 'seed',
              'runtime_profile', 'context_limit', 'task_mode', 'loss_reduction', 'total_updates')
    for field in fields:
        if field not in previous or current.get(field) != previous[field]:
            raise ValueError('candidate continuation changed ' + field)
    start = current.get('start_update')
    count = current.get('updates')
    if (type(start) is not int or type(count) is not int or start < 1 or count < 1
            or start != previous.get('start_update', 0) + previous['updates']
            or start != state['updates'] or start + count > current['total_updates']):
        raise ValueError('exact bounded candidate predecessor cursor required')
    if (state['training_root'] != current['previous_root']
            or current['initial_adapter'] != state['adapter']
            or state['candidate_profile'] != current['candidate_profile']):
        raise ValueError('exact accepted candidate adapter and predecessor required')


def authenticate(config):
    def read(path):
        path = Path(path)
        if config['pins'].get(str(path)) != file_hash(path):
            raise ValueError('unpinned candidate evidence: ' + str(path))
        return json.loads(path.read_text())

    previous = read(Path(config['previous_root']) / 'config.json')
    state = read(config['candidate_state_acceptance'])
    gate = read(config['candidate_parity_acceptance'])
    if any(state.get(key) is not True for key in
           ('passed', 'exact_master_export', 'exact_peft_reload', 'durable_commit_verified',
            'exact_length_permutation')):
        raise ValueError('complete independently verified candidate state required')
    for path, digest in state['files'].items():
        if config['pins'].get(path) != digest or file_hash(path) != digest:
            raise ValueError('accepted candidate state artifact changed')
    if (state.get('candidate_profile') != config['candidate_profile']
            or not gate.get('passed') or gate.get('requests') != 256
            or gate.get('adapter_id') != state['adapter_id']
            or gate.get('state_sha256') != file_hash(config['candidate_state_acceptance'])
            or gate.get('candidate_profile') != config['candidate_profile']):
        raise ValueError('same-checkpoint complete candidate backend gate required')
    for path, digest in gate['files'].items():
        if config['pins'].get(path) != digest or file_hash(path) != digest:
            raise ValueError('accepted candidate backend artifact changed')
    validate_lineage(previous, config, state)
    return dict(adapter_id=state['adapter_id'], completed_updates=state['updates'],
                state_sha256=file_hash(config['candidate_state_acceptance']),
                parity_sha256=file_hash(config['candidate_parity_acceptance']))


def check_restored_policy(model, processor, config, output, *, rank, device):
    import torch
    import torch.distributed as dist
    from .simple_prepared_inputs import prepared
    from .span_audio_collate import forward_span_audio_batch
    from .span_parity import compare_span_policies

    gate = json.loads(Path(config['candidate_parity_acceptance']).read_text())
    paths = [gate['backend_config'], gate['inputs'], gate['hf_results']]
    for path in paths:
        if config['pins'].get(path) != file_hash(path):
            raise ValueError('pinned recovery input and HF result required')
    backend = json.loads(Path(gate['backend_config']).read_text())
    artifact = json.loads(Path(gate['inputs']).read_text())
    hf = json.loads(Path(gate['hf_results']).read_text())
    if hf['adapter_id'] != gate['adapter_id'] or backend['adapter_id'] != gate['adapter_id']:
        raise ValueError('recovered candidate policy differs')
    references = {r['id']: r for r in hf['records']}
    rows = list(prepared(backend, artifact, processor=processor))[rank::4]
    if len(rows) != 64 or len(references) != 256:
        raise ValueError('complete synchronized recovery population required')
    collation = config['candidate_collation']
    results = []
    with torch.no_grad():
        for micro in (1, config['microbatch_per_rank']):
            for start in range(0, len(rows), micro):
                group = rows[start:start + micro]
                payload = [dict(inputs=r[1], positions=r[2], candidates=r[3]) for r in group]
                probabilities = forward_span_audio_batch(model, payload, device=device,
                                                         dtype=torch.bfloat16, **collation)
                for row, probs in zip(group, probabilities, strict=True):
                    ref = references[row[0]['id']]
                    if ref['request_id'] != row[4] or ref['request_identity'] != row[5]:
                        raise ValueError('recovered actor request differs')
                    logp = [p[0].cpu().tolist() for p in probs]
                    results.append(dict(id=row[0]['id'], microbatch=micro, candidate_logprobs=logp,
                                        parity=compare_span_policies(ref['candidate_logprobs'], logp)))
    local = dict(rank=rank, passed=len(results) == 128 and all(r['parity']['passed'] for r in results),
                 requests=64, comparisons=len(results),
                 maximum_logprob_error=max(r['parity']['maximum_logprob_error'] for r in results))
    peers = [None] * 4
    dist.all_gather_object(peers, local)
    atomic_json(output/'restored-policy.json', dict(**local, records=results, ranks=peers,
        scope='Restored exact FP32 state, single and batched resident candidate vs accepted HF256 before update.'))
    if any(not p['passed'] for p in peers):
        raise ValueError('recovered candidate policy failed full probability gate')
    return local
