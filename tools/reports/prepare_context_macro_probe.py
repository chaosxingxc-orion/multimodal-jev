"""Seal dual-service configs and a complete macro128 measurement payload.

Preparation only. Reuses authenticated real inputs/policy and a successful
independently audited serial/batch probe. No devices or running jobs touched.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.local_config import load_config
from tools.reports.context_macro_measurement import check_reference
from tools.reports.context_dual_runtime_candidate import RUNTIME, WIRE
from tools.reports.run_context_batch_probe_candidate import verify

ADDITIONS = (
    'src/multimodal_jev/local_config.py',
    'src/multimodal_jev/context_macro_rollout.py',
    'tools/reports/context_macro_client.py',
    'tools/reports/context_macro_measurement.py',
    'tools/reports/context_dual_runtime_candidate.py',
    'tools/reports/serve_context_dual_candidate.py',
    'tools/reports/serve_context_batched_candidate.py',
)


def prepare(c, provenance):
    parent = Path(c['parent']); source = Path(c['source']); output = Path(c['output'])
    receipt = json.loads((parent/'PREPARED.json').read_text())
    if verify(parent/'package') != c['parent_manifest_sha256'] or receipt['files_sha256'] != c['parent_manifest_sha256']:
        raise ValueError('sealed parent payload changed')
    if file_hash(c['parent_audit']) != c['parent_audit_sha256']:
        raise ValueError('parent audit changed')
    audit = json.loads(Path(c['parent_audit']).read_text())
    if (audit['status'] != 'finite_fixed_policy_inference_audited'
            or audit['files_sha256'] != receipt['files_sha256']
            or audit['exact_actual_logits'] is not True
            or audit['exact_probabilities_and_G16'] is not True):
        raise ValueError('successful actual logits parent audit required')
    reference_path = Path(c['serial_reference'])
    if file_hash(reference_path) != c['serial_reference_sha256']:
        raise ValueError('serial measurement changed')
    # The reference must be the file covered by the exact parent transport.
    transport = Path(c['parent_transport'])
    if file_hash(transport) != audit['transport_sha256']:
        raise ValueError('parent transport binding differs')
    transported = json.loads(transport.read_text())
    if transported['files']['evaluation/serial/measurement.json'] != c['serial_reference_sha256']:
        raise ValueError('reference not covered by parent audit')
    original = json.loads((parent/'package/config.json').read_text())
    rows = torch.load(parent/'package/inputs.pt', map_location='cpu', weights_only=True)
    check_reference(json.loads(reference_path.read_text()), rows,
                    original['policy']['policy_id'], original['input_sha256'])
    remote = Path(c['remote_root']); ipc = Path(c['ipc_root'])
    if (not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or not ipc.is_absolute() or '..' in ipc.parts or len(str(ipc).encode())+6 > 107
            or output.exists() or output.resolve().is_relative_to(parent.resolve())
            or str(remote) == receipt['remote_root']):
        raise ValueError('new explicit local output/runtime and short IPC roots required')
    output.mkdir(parents=True); package = output/'package'
    shutil.copytree(parent/'package', package)
    (package/'FILES.json').unlink(); (package/'config.json').unlink()
    for name in ADDITIONS:
        target = package/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    shutil.copy2(reference_path, package/'serial-reference.json')
    shutil.copy2(c['parent_audit'], package/'parent-audit.json')
    # The manifest covers the complete service Python source set.
    sources = {str(p.relative_to(package)): file_hash(p) for p in package.rglob('*.py')}
    (package/'RUNTIME-SOURCES.json').write_text(json.dumps(sources, sort_keys=True, indent=2)+'\n')
    runtime = dict(image_sha256=receipt['image'].removeprefix('sha256:'),
        source_manifest_sha256=file_hash(package/'RUNTIME-SOURCES.json'),
        processor_sha256=processor_cache_identity(package/'policy'), profile=RUNTIME, output_protocol=WIRE)
    runtime_root = str(remote/'package')
    pins = {runtime_root+'/'+str(p.relative_to(package)):file_hash(p)
            for p in package.rglob('*') if p.is_file()}
    policy = dict(original['policy'], model=c['model'], snapshot=runtime_root+'/policy',
                  adapter=runtime_root+'/adapter')
    replicas = []
    for index, device in enumerate((6, 7)):
        endpoint = dict(socket=str(ipc/f'r{index}.sock'), exchange=str(ipc/f'exchange-{index}'))
        replicas.append(endpoint)
        service = dict(source_root=runtime_root, source_policy=policy['snapshot'],
            dual_runtime_manifest=runtime_root+'/RUNTIME-SOURCES.json', dual_runtime_identity=runtime,
            transport_protocol='bulk_dual_logits_candidate_v1', pins=pins, **endpoint)
        service['socket_path'] = service.pop('socket')
        service['policy'] = dict(policy, physical_devices=[device])
        service['policy'].pop('process_device_scope', None)
        (package/f'service-{index}.json').write_text(json.dumps(service,indent=2)+'\n')
    client = dict(schema='jev-macro-client-v1', attempt_id='finite-audio-v1',
        journal_root=str(remote/'journal'), runtime_identity=runtime, replicas=replicas,
        timeout_seconds=900, rollout_queries=32, max_queries=128)
    measurement = dict(output=str(remote/'measurement'), pins=copy.deepcopy(pins),
        inputs=runtime_root+'/inputs.pt', serial_reference=runtime_root+'/serial-reference.json',
        policy=policy, macro_client=client, audio_token_id=original['audio_token_id'])
    (package/'measurement.json').write_text(json.dumps(measurement,indent=2)+'\n')
    files = {str(p.relative_to(package)):file_hash(p) for p in package.rglob('*') if p.is_file()}
    (package/'FILES.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
    result = dict(status='macro_probe_prepared_only', image=receipt['image'],
        files_sha256=verify(package), parent_files_sha256=receipt['files_sha256'],
        parent_audit_sha256=c['parent_audit_sha256'], remote_root=str(remote), ipc_root=str(ipc),
        runtime_identity=runtime, config_provenance=provenance,
        device_scope=[[6],[7]], training_dispatch_allowed=False,
        pending=['isolated two-service launcher/image-device-scope inspection',
                 'actual full macro and failure/closure audit', 'DP2 actor integration'])
    (output/'PREPARED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--local',required=True)
    args=parser.parse_args(); config,provenance=load_config(args.config,args.local)
    print(json.dumps(prepare(config,provenance),indent=2))
