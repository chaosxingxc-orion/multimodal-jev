#!/usr/bin/env python3
"""Independently recompute prefix or complete selection HF/vLLM evidence."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash, atomic_json
from multimodal_jev.span_parity import compare_span_policies
from multimodal_jev.policy_identity import checkpoint_identity


def verify(config_path):
    audit_config=json.loads(Path(config_path).read_text())
    for name,digest in audit_config['pins'].items():
        if file_hash(name)!=digest:raise ValueError('candidate parity audit dependency changed')
    root = Path(audit_config['backend_root'])
    files = {str(config_path): file_hash(config_path), **audit_config['pins']}
    def read(path):
        path = Path(path)
        files[str(path)] = file_hash(path)
        return json.loads(path.read_text())
    config = read(root/'config.json')
    launch = read(audit_config['backend_launch'])
    observed = read(audit_config['backend_observed'])
    if (launch['config_sha256'] != file_hash(root/'config.json')
            or observed['container_id'] != launch['container_id'] or observed['image'] != launch['image']
            or observed['state']['Status'] != 'exited' or observed['state']['ExitCode'] != 0):
        raise ValueError('exact successfully completed parity container required')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('parity dependency changed: '+path)
        files[path] = digest
    state_path = Path(audit_config['checkpoint_evidence'])
    training_observed_path = Path(audit_config['training_observed'])
    for key, path in [('checkpoint_evidence', state_path), ('training_observed', training_observed_path)]:
        if audit_config['pins'].get(str(path)) != file_hash(path):
            raise ValueError('explicit checkpoint evidence must be pinned')
    state = read(state_path)
    training_observed = read(training_observed_path)
    training_root = Path(state['training_root'])
    training_launch = read(audit_config['training_launch'])
    training_config = read(training_root/'config.json')
    if (training_observed['container_id'] != training_launch['container_id']
            or training_observed['image'] != training_launch['image']
            or training_observed['state']['Status'] != 'exited'
            or training_observed['state']['ExitCode'] != 0
            or training_launch['config_sha256'] != file_hash(training_root/'config.json')
            or training_config.get('task_mode') != 'direct_final'
            or not all(state.get(k) is True for k in
                       ('passed', 'exact_master_export', 'exact_peft_reload', 'durable_commit_verified'))
            or state['adapter_id'] != config['adapter_id'] or state['original_adapter_id'] != config['adapter_id']):
        raise ValueError('accepted durable simple-Final checkpoint required')
    for path, digest in state['files'].items():
        if file_hash(path) != digest:
            raise ValueError('independent state evidence changed')
        files[path] = digest
    if checkpoint_identity(config['adapter'])[0] != config['adapter_id']:
        raise ValueError('loaded adapter changed')
    from multimodal_jev.span_batch_training_config import PROFILE
    if (training_config.get('candidate_profile')!=PROFILE or config.get('candidate_component')!='full'
            or config.get('rollout_candidate_component')!='full_native_lora'):
        raise ValueError('explicit complete candidate profiles required')
    artifact = read(config['inputs'])
    mode = artifact.get('population_mode', 'prefix')
    if mode == 'full_selection':
        from multimodal_jev.slurp_official_training import load
        from multimodal_jev.simple_selection_inputs import verify as verify_inputs
        if (state['updates'] != 634 or training_config.get('total_updates') != 634
                or state.get('epoch') != 2 or state.get('next_batch') != 0
                or training_config.get('base_id') != config['base_id']
                or not training_config.get('runtime_profile')
                or training_config['runtime_profile'] != config.get('runtime_profile')):
            raise ValueError('complete terminal634 same-profile policy required for selection')
        rows, task, _ = load(training_config['release'], training_config['release_sha256'],
                             'selection', purpose='evaluate')
        verify_inputs(artifact, rows, task, training_config['release_sha256'])
        count = len(rows)
        if count != 9226:
            raise ValueError('frozen full9226 selection required')
        manifest = str(Path(training_config['release'])/'MANIFEST.json')
        files[manifest] = file_hash(manifest)
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
    return dict(passed=True,training_root=str(training_root),updates=state['updates'],
        state_sha256=file_hash(state_path),adapter_id=config['adapter_id'],adapter=config['adapter'],
        candidate_profile=PROFILE,backend_config=str(root/'config.json'),inputs=config['inputs'],
        hf_results=str(root/'run/hf/result.json'),
        full_candidate_parity=True,exact_master_export=True,requests=count,population_mode=mode,
        maximum_logprob_error=max(c['maximum_logprob_error'] for c in checks),
        argmax_changed_spans=sum(len(c['argmax_changed_spans']) for c in checks),files=files,
        sampled_resident_parity=None,
        scope='Independent complete declared population HF/vLLM parity plus durable master/export/reload evidence; resident recovery and task metrics separate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, verify(args.config))
