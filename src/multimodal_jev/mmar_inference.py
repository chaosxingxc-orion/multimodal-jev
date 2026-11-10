"""Frozen MMAR generation contract and admission of complete processor inputs."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .baseline import file_hash
from .mmar_omni import INPUT_SHA, INSTRUCTION

PROCESSOR_RESULT_SHA = '7b13da4e5cdb91918f8fbd9da772e3b778be295d620b67c76c1decc9bde26d53'
IMAGE = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
ENGINE = dict(dtype='bfloat16', tensor_parallel_size=2, seed=17, max_model_len=4096,
              max_num_seqs=1, max_num_batched_tokens=4096, gpu_memory_utilization=.85,
              enforce_eager=True, trust_remote_code=False,
              limit_mm_per_prompt={'audio': 1, 'image': 0, 'video': 0}, enable_prefix_caching=False)


def ids_hash(ids):
    # Match the original accepted processor run, not canonical JSON serialization.
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def population(inputs, processor_run):
    root = Path(processor_run)
    if file_hash(inputs) != INPUT_SHA or file_hash(root/'result.json') != PROCESSOR_RESULT_SHA:
        raise ValueError('registered complete MMAR inputs and processor result required')
    result = json.loads((root/'result.json').read_text())
    if (result['status'] != 'passed' or result['records'] != 1000
            or json.loads((root/'terminal.json').read_text())['exit_code'] != 0
            or result['identity_sha256'] != file_hash(root/'identity.json')
            or result['records_sha256'] != file_hash(root/'records.jsonl')):
        raise ValueError('complete unmodified processor evidence required')
    identity = json.loads((root/'identity.json').read_text())
    if (identity['inputs_sha256'] != INPUT_SHA or identity['instruction'] != INSTRUCTION
            or identity['max_model_len'] != 4096 or identity['reserved_output_tokens'] != 256
            or identity['targets_read'] or identity['model_loaded']):
        raise ValueError('original processor contract required')
    rows = [json.loads(line) for line in Path(inputs).open()]
    prior = [json.loads(line) for line in (root/'records.jsonl').open()]
    if (len(rows) != 1000 or len(prior) != 1000 or len({r['id'] for r in rows}) != 1000
            or [r['id'] for r in rows] != [r['id'] for r in prior]):
        raise ValueError('complete original ordered population required')
    return rows, prior, identity


def capacity_indices(prior):
    """Input-only maxima, ties resolved by original order; no answer selection."""
    return sorted({max(range(len(prior)), key=lambda i: prior[i]['input_tokens']),
                   max(range(len(prior)), key=lambda i: prior[i]['audio']['actor_samples'])})


def validate_input(request, receipt, ids, frames, prior):
    if (receipt != prior['audio'] or frames != prior['frames']
            or hashlib.sha256(request['prompt'].encode()).hexdigest() != prior['prompt_sha256']
            or ids_hash(ids) != prior['input_ids_sha256'] or len(ids) != prior['input_tokens']
            or len(ids)+256 > 4096):
        raise ValueError('actual input differs from accepted full processor run')


def prepare_checked(processor, row, audio_root, prior):
    """Shared real input path for capacity, full inference and CPU preflight."""
    import torch
    from .mmar_omni import prepare_request
    request, receipt = prepare_request(processor, row, audio_root)
    wave = request['multi_modal_data']['audio'][0][0]
    batch = processor(text=[request['prompt']], audio=[wave], sampling_rate=16000,
                      return_tensors='pt', padding=True, truncation=False)
    ids = batch['input_ids'][0].tolist()
    validate_input(request, receipt, ids, batch['feature_attention_mask'].sum(-1).tolist(), prior)
    if not torch.isfinite(batch['input_features']).all():
        raise ValueError('nonfinite audio features')
    return request, receipt, ids


def accept_response(responses, ids, budget):
    if len(responses) != 1 or not responses[0].finished or len(responses[0].outputs) != 1:
        raise ValueError('one completed generation required')
    response = responses[0]; answer = response.outputs[0]
    tokens = list(answer.token_ids)
    if list(response.prompt_token_ids) != ids:
        raise ValueError('actual engine input IDs differ')
    if (type(answer.text) is not str or answer.finish_reason not in ('stop', 'length')
            or not 1 <= len(tokens) <= budget
            or any(type(t) is not int or t < 0 for t in tokens)
            or (answer.finish_reason == 'length' and len(tokens) != budget)):
        raise ValueError('generation token budget or termination differs')
    return dict(raw_text=answer.text, token_ids=tokens, finish_reason=answer.finish_reason,
                stop_reason=answer.stop_reason, length_limited=answer.finish_reason=='length')


def admit_capacity(root, intent_hash, inspection, inspection_hash, rows, prior):
    """Caller must freshly inspect the exact container and pin the saved JSON.

    A supplied file alone is not proof of external container provenance.
    """
    root = Path(root)
    if file_hash(root/'intent.json') != intent_hash or file_hash(inspection) != inspection_hash:
        raise ValueError('pinned capacity intent and actual inspection required')
    plan = json.loads((root/'intent.json').read_text())
    launch = json.loads((root/'launch.json').read_text())
    item = json.loads(Path(inspection).read_text()); state = item['State']; host = item['HostConfig']
    argv = plan['argv']
    expected_devices = {argv[i+1].split(':')[0] for i,v in enumerate(argv[:-1]) if v=='--device'}
    expected_binds = {argv[i+1] for i,v in enumerate(argv[:-1]) if v=='-v'}
    expected_env = {argv[i+1] for i,v in enumerate(argv[:-1]) if v=='-e'}
    if (launch['intent_sha256'] != intent_hash or item['Id'] != launch['container_id']
            or item['Image'] != plan['image'] or plan['image'] != IMAGE
            or state['Status'] != 'exited' or state['ExitCode'] != 0
            or state['OOMKilled'] or state['Error'] or state['Running']
            or host['Privileged'] or host['NanoCpus'] != 16*10**9 or host['Memory'] != 128*1024**3
            or host['ShmSize'] != 8*1024**3 or host['NetworkMode'] != 'host'
            or {d['PathOnHost'] for d in host['Devices']} != expected_devices
            or not {'/dev/davinci6', '/dev/davinci7'}.issubset(expected_devices)
            or any('/dev/davinci'+str(i) in expected_devices for i in range(6))
            or set(host['Binds']) != expected_binds or not expected_env.issubset(set(item['Config']['Env']))
            or item['Config']['Entrypoint'] != ['/opt/multimodal-jev/runtime-entrypoint.sh']
            or item['Config']['Cmd'] != argv[argv.index(plan['image'])+1:] or '--capacity-only' not in argv):
        raise ValueError('exact successful NPU6/7 capacity container required')
    for name,pin in plan['source_files'].items():
        if file_hash(root/'source'/name) != pin:
            raise ValueError('capacity source changed')
    run = root/'run'; result = json.loads((run/'result.json').read_text())
    identity = json.loads((run/'identity.json').read_text())
    from .audio2tool_engine import ASSET_AUDIT_SHA, VERSIONS
    audit = json.loads((root/'asset-audit.json').read_text())
    if (file_hash(root/'asset-audit.json') != ASSET_AUDIT_SHA
            or identity['asset_audit_sha256'] != ASSET_AUDIT_SHA
            or identity['model'] != audit['model'] or identity['model_files'] != audit['files']
            or identity['versions'] != VERSIONS):
        raise ValueError('capacity runtime or model differs')
    current = Path(__file__).resolve().parents[2]
    for name,pin in identity['source_files'].items():
        if plan['source_files'].get(name) != pin:
            raise ValueError('capacity runtime source differs from snapshot')
    for name in ('src/multimodal_jev/mmar_omni.py', 'src/multimodal_jev/mmar_inference.py',
                 'src/multimodal_jev/sfc_audio.py', 'src/multimodal_jev/omni_waveform_boundary.py',
                 'tools/jev/infer_mmar_omni.py'):
        if identity['source_files'].get(name) != file_hash(current/name):
            raise ValueError('capacity and full inference implementations differ')
    indices = capacity_indices(prior)
    if (json.loads((run/'terminal.json').read_text())['exit_code'] != 0
            or result['status'] != 'passed' or result['mode'] != 'capacity' or result['weight_updates'] != 0
            or result['records'] != len(indices) or result['targets_read'] or not result['model_files_unchanged']
            or result['identity_sha256'] != file_hash(run/'identity.json')
            or result['predictions_sha256'] != file_hash(run/'predictions.jsonl')
            or identity['engine'] != ENGINE or identity['sampling'] != dict(temperature=0.,max_tokens=1,n=1,seed=17)
            or identity['inputs_sha256'] != INPUT_SHA or identity['processor_result_sha256'] != PROCESSOR_RESULT_SHA
            or identity['selected_ids'] != [rows[i]['id'] for i in indices]
            or not identity['parameters_frozen'] or identity['targets_read']):
        raise ValueError('actual capacity inference required')
    records = [json.loads(line) for line in (run/'predictions.jsonl').open()]
    if len(records) != len(indices):
        raise ValueError('all deterministic capacity boundaries required')
    names = {f'{i:04d}.json' for i in indices}
    if ({p.name for p in (run/'attempts').iterdir()} != names
            or {p.name for p in (run/'records').iterdir()} != names
            or any((run/'failures').iterdir())):
        raise ValueError('complete successful capacity attempt ledger required')
    for r,i in zip(records,indices,strict=True):
        if (r['id'] != rows[i]['id'] or r['audio'] != prior[i]['audio']
                or r['input_ids_sha256'] != prior[i]['input_ids_sha256']
                or r['prompt_sha256'] != prior[i]['prompt_sha256']
                or len(r['token_ids']) != 1 or r['finish_reason'] not in ('stop','length')
                or r.get('error') is not None):
            raise ValueError('capacity boundary differs')
        attempt = json.loads((run/'attempts'/f'{i:04d}.json').read_text())
        raw = json.loads((run/'records'/f'{i:04d}.json').read_text())
        if (raw['attempt'] != attempt or any(r.get(k) != v for k,v in attempt.items())
                or attempt['request_sha256'] != hashlib.sha256(json.dumps(rows[i],sort_keys=True).encode()).hexdigest()
                or not math.isfinite(r['seconds']) or not 0 <= raw['seconds'] <= r['seconds']):
            raise ValueError('capacity raw response or attempt differs')
        responses = [SimpleNamespace(**{**v, 'outputs': [SimpleNamespace(**a) for a in v['outputs']]})
                     for v in raw['responses']]
        if len(responses) != 1 or ids_hash(responses[0].prompt_token_ids) != prior[i]['input_ids_sha256']:
            raise ValueError('capacity raw engine IDs differ')
        accepted = accept_response(responses, responses[0].prompt_token_ids, 1)
        if any(r.get(k) != v for k,v in accepted.items()):
            raise ValueError('capacity summary differs from raw generation')
    return dict(container_id=item['Id'], intent_sha256=intent_hash,
                inspection_sha256=inspection_hash, result_sha256=file_hash(run/'result.json'))
