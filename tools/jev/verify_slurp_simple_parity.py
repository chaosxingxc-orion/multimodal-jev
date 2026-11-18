#!/usr/bin/env python3
"""Independently recompute prefix or complete selection HF/vLLM evidence."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash, atomic_json
from multimodal_jev.span_parity import compare_span_policies
from multimodal_jev.policy_identity import checkpoint_identity


def verify(root):
    root = Path(root)
    files = {}
    def read(path):
        path = Path(path)
        files[str(path)] = file_hash(path)
        return json.loads(path.read_text())
    config = read(root/'config.json')
    launch = read(root/'launch.json')
    observed = read(root/'observed-final.json')
    if (launch['config_sha256'] != file_hash(root/'config.json')
            or observed['container_id'] != launch['container_id'] or observed['image'] != launch['image']
            or observed['state']['Status'] != 'exited' or observed['state']['ExitCode'] != 0):
        raise ValueError('exact successfully completed parity container required')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('parity dependency changed: '+path)
        files[path] = digest
    state_path = Path(config.get('checkpoint_evidence', root/'state-verified.json'))
    training_observed_path = Path(config.get('training_observed', root/'training-observed.json'))
    for key, path in [('checkpoint_evidence', state_path), ('training_observed', training_observed_path)]:
        if key in config and config['pins'].get(str(path)) != file_hash(path):
            raise ValueError('explicit checkpoint evidence must be pinned')
    state = read(state_path)
    training_observed = read(training_observed_path)
    training_root = Path(state['training_root'])
    training_launch = read(training_root/'launch.json')
    training_config = read(training_root/'config.json')
    if (training_observed['container_id'] != training_launch['container_id']
            or training_observed['image'] != training_launch['image']
            or training_observed['state']['Status'] != 'exited'
            or training_observed['state']['ExitCode'] != 0
            or training_launch['config_sha256'] != file_hash(training_root/'config.json')
            or training_config.get('task_mode') != 'direct_final'
            or not all(state.get(k) is True for k in
                       ('passed', 'exact_master_export', 'exact_peft_reload', 'durable_commit_verified'))
            or state['adapter_id'] != config['adapter_id'] or state['adapter'] != config['adapter']):
        raise ValueError('accepted durable simple-Final checkpoint required')
    for path, digest in state['files'].items():
        if file_hash(path) != digest:
            raise ValueError('independent state evidence changed')
        files[path] = digest
    if checkpoint_identity(config['adapter'])[0] != config['adapter_id']:
        raise ValueError('loaded adapter changed')
    artifact = read(config['inputs'])
    mode = artifact.get('population_mode', 'prefix')
    if mode == 'full_selection':
        from multimodal_jev.slurp_official_training import load
        from multimodal_jev.simple_selection_inputs import verify as verify_inputs
        if (state['updates'] != training_config['total_updates']
                or state['updates'] != 634 or state.get('epoch') != 2 or state.get('next_batch') != 0
                or training_config.get('base_id') != config['base_id']
                or training_config.get('runtime_profile') != config.get('runtime_profile')
                or 'checkpoint_evidence' not in config or 'training_observed' not in config):
            raise ValueError('complete terminal634 same-profile policy required for selection')
        rows, task, _ = load(training_config['release'], training_config['release_sha256'],
                             'selection', purpose='evaluate')
        verify_inputs(artifact, rows, task, training_config['release_sha256'])
        count = len(rows)
        if count != 9226:
            raise ValueError('frozen full9226 selection required')
    elif mode == 'prefix':
        count = 256
    else:
        raise ValueError('unknown parity population mode')
    if artifact['requests'] != count or artifact['recordings'] != count or not artifact['passed']:
        raise ValueError('complete independent input population required')
    inputs = {r['id']: r for r in artifact['records']}
    if len(inputs) != count or len(artifact['records']) != count:
        raise ValueError('unique complete request identities required')
    backends = {}
    for backend in ('hf', 'vllm'):
        if read(root/'run'/backend/'terminal.json')['exit_code'] != 0:
            raise ValueError('backend failed')
        result = read(root/'run'/backend/'result.json')
        if (not result['passed'] or result['backend'] != backend
                or result['base_id'] != config['base_id'] or result['adapter_id'] != config['adapter_id']):
            raise ValueError('backend policy differs')
        records = {r['id']: r for r in result['records']}
        if len(result['records']) != count or records.keys() != inputs.keys():
            raise ValueError('complete backend request population required')
        backends[backend] = records
    checks = []
    for rid, row in inputs.items():
        hf, vllm = (backends[b][rid] for b in ('hf', 'vllm'))
        identity = hf['request_identity']
        import hashlib
        digest = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if (hf['request_id'] != digest or vllm['request_id'] != digest
                or vllm['request_identity'] != identity or identity['request'] != row['request']
                or identity['policy_id'] != config['base_id']+':'+config['adapter_id']
                or identity['audio_processing'] != row['audio_processing']
                or identity['positions'] != row['positions']
                or identity['candidate_ids'] != [m['token_ids'] for m in row['candidate_mapping']]
                or any(identity['inputs'][k] != row[k] for k in
                       ('input_ids','input_features','input_features_mask'))):
            raise ValueError('exact independent input binding differs')
        checks.append(compare_span_policies(hf['candidate_logprobs'],vllm['candidate_logprobs']))
    if read(root/'run/terminal.json')['exit_code'] != 0 or not all(c['passed'] for c in checks):
        raise ValueError('independent full-candidate probability check failed')
    sampled=None
    if 'sampled_probe' in training_config:
        from multimodal_jev.simple_sampled_parity import compare_resident
        second=training_config['sampled_probe'].get('fixture_initialization')=='restore_updated_rl_then_second_update'
        cursor=2 if second else 1
        if (mode!='prefix' or state['updates']!=cursor or state.get('supervision_records_verified')!=128
                or state.get('algorithm')!=training_config['sampled_probe']['algorithm']
                or config.get('runtime_profile')!=training_config['runtime_profile']
                or config['base_id']!=training_config['base_id']
                or 'checkpoint_evidence' not in config or 'training_observed' not in config):
            raise ValueError('independently audited sampled one-update policy required')
        inputs_path=training_root/'run/rl-inputs.json'
        if str(inputs_path) not in state['files']:
            raise ValueError('independently audited sampling required')
        sampling=read(inputs_path)['sampling']
        behavior_id=state['behavior_policy_id'] if second else state['reference_policy_id']
        if (sampling['policy_id']!=behavior_id or sampling['reference_policy_id']!=state['reference_policy_id']
                or sampling['update']!=cursor-1):
            raise ValueError('sampled behavior/reference/cursor lineage differs')
        resident=[]
        for rank in range(4):
            path=training_root/'run'/f'rl-rank-{rank}/after.json'
            if str(path) not in state['files']:raise ValueError('audited resident outputs required')
            rows=read(path)['records']
            if len(rows)!=32:raise ValueError('complete per-rank resident outputs required')
            resident.extend(rows)
        sampled={b:compare_resident(resident,list(backends[b].values()),sampling['sampling'],
            old_policy_id=behavior_id,new_policy_id=config['base_id']+':'+config['adapter_id'],update=cursor-1)
            for b in ('hf','vllm')}
    return dict(passed=True,training_root=str(training_root),updates=state['updates'],
        state_sha256=file_hash(state_path),adapter_id=config['adapter_id'],adapter=config['adapter'],
        full_candidate_parity=True,exact_master_export=True,requests=count,population_mode=mode,
        maximum_logprob_error=max(c['maximum_logprob_error'] for c in checks),
        argmax_changed_spans=sum(len(c['argmax_changed_spans']) for c in checks),files=files,
        sampled_resident_parity=sampled,
        scope='Independent complete declared population HF/vLLM parity plus durable master/export/reload evidence; resident recovery and task metrics separate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    output = args.root/'verified.json'
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, verify(args.root))
