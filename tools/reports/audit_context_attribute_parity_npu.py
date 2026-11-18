"""Offline independent audit of the finite attribute HF/vLLM diagnostic.

Bind actual container, collected bytes and both backend input identities to
previously executed CPU anchors. No remote calls or inference in this tool.
"""
import argparse
import json
from pathlib import Path

from tools.reports.launch_context_attribute_parity_npu_v2 import validate
from tools.reports.prepare_context_attribute_parity_npu_v2 import sha
from tools.reports.probe_context_attribute_parity_v2 import compare, REFERENCE_ID
from tools.reports.build_context_attribute_candidate import request, digest


def read(path):
    return json.loads(Path(path).read_text())


def bind_inputs(config, menu, anchors, hf, vl):
    """Both backends agreeing is insufficient: bind to independent CPU inputs."""
    if anchors['policy_id'] != REFERENCE_ID or len(anchors['inputs']) != 6:
        raise ValueError('Complete frozen CPU anchor evidence required')
    expected = []
    for arm in ('direct', 'q_input'):
        for asset in config['audio']:
            req = request(asset['audio'], menu, arm=arm)
            expected.append(dict(id=asset['id'], arm=arm,
                request_sha256=digest(req.model_input()), decision_ids=[q.id for q in req.questions]))
    if hf['input_panel'] != expected or vl['input_panel'] != expected:
        raise ValueError('Actual request panel differs from pinned audio/catalog')
    for anchor, entry, a, b in zip(anchors['inputs'], expected, hf['records'], vl['records'], strict=True):
        if any(anchor[k] != entry[k] for k in ('id','arm','decision_ids')):
            raise ValueError('CPU anchor order differs')
        for record in (a, b):
            identity = record['request_identity']
            if (identity['inputs'] != anchor['tensors']
                    or identity['positions'] != anchor['positions'][0]):
                raise ValueError('Actual model input differs from independently prepared tensors')
    return compare(hf, vl)


def check_runtime(receipt, config, launch, observed):
    argv = receipt['argv']
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v == flag]
    state = observed['State']
    if (launch['argv'] != argv or launch['image'] != receipt['image']
            or launch['remote_root'] != receipt['remote_root']
            or launch['source_manifest_sha256'] != receipt['source_manifest_sha256']
            or observed['Id'] != launch['container_id'] or observed['Image'] != receipt['image']
            or state['Status'] != 'exited' or state['ExitCode'] != 0 or state['OOMKilled']
            or observed['Config']['Entrypoint'] != values('--entrypoint')
            or observed['Config']['Cmd'] != argv[argv.index(receipt['image'])+1:]
            or sorted(observed['HostConfig']['Binds']) != sorted(values('-v'))
            or observed['HostConfig']['NetworkMode'] != 'none'):
        raise ValueError('Exact successful isolated runtime required')
    environment = dict(v.split('=',1) for v in observed['Config']['Env'])
    if any(environment.get(k) != v for k,v in (e.split('=',1) for e in values('-e'))):
        raise ValueError('Runtime environment differs')
    devices = observed['HostConfig']['Devices']
    if sorted((d['PathOnHost'],d['PathInContainer'],d['CgroupPermissions']) for d in devices) != sorted(
            (*v.split(':'),'rwm') for v in values('--device')):
        raise ValueError('Actual device scope differs')
    if config['physical_devices'] != [7]:raise ValueError('Frozen device required')


def audit(prepared, launch, observed, collection, transport, cpu_evidence, cpu_collection):
    prepared, collection, cpu_collection = map(Path, (prepared, collection, cpu_collection))
    receipt, config = validate(prepared)
    check_runtime(receipt, config, launch, observed)
    if launch['config_sha256'] != sha(prepared/'package/probe-config.json'):
        raise ValueError('Launched configuration differs')
    files = {}
    for path in collection.rglob('*'):
        if path.is_symlink():raise ValueError('Collection symlinks forbidden')
        if path.is_file():files[str(path.relative_to(collection))] = sha(path)
    if (files != transport['files'] or files['config.json'] != launch['config_sha256']
            or transport['status'] != 'transport_complete_independent_audit_required'
            or transport['container_id'] != launch['container_id']
            or transport['image'] != receipt['image'] or transport['remote_root'] != receipt['remote_root']
            or transport['state'] != observed['State']):
        raise ValueError('Complete authenticated transport required')
    proof = read(cpu_evidence)
    if (proof['image'] != receipt['image'] or proof['state']['Status'] != 'exited'
            or proof['state']['ExitCode'] != 0 or proof['state']['OOMKilled']
            or proof['counts'] != dict(tests=14,failures=0,errors=0,skipped=0)):
        raise ValueError('Successful v2 probe CPU preparation required')
    for name, h in proof['files'].items():
        if sha(cpu_collection/name) != h:raise ValueError('CPU evidence payload changed')
    if (sha(cpu_collection/'package/FILES.json') != proof['source_manifest_sha256']
            or read(cpu_collection/'run/terminal.json')['source_manifest_sha256'] != proof['source_manifest_sha256']
            or read(cpu_collection/'run/terminal.json')['exit_code'] != 0):
        raise ValueError('CPU terminal and executed source manifest differ')
    cpu_files = read(cpu_collection/'package/FILES.json')
    for name,h in read(prepared/'package/FILES.json').items():
        if name not in ('probe-config.json','run.py') and cpu_files.get(name) != h:
            raise ValueError('NPU payload differs from executed CPU sources/assets')
    anchors = read(cpu_collection/'run/prepared.json')
    menu = read(prepared/'package/input-fixture/catalog.json')
    root = collection/'evaluation'
    terminal = read(root/'terminal.json')
    if terminal['exit_code'] != 0 or terminal['source_manifest_sha256'] != receipt['source_manifest_sha256']:
        raise ValueError('Successful bound terminal required')
    for name in ('hf-runtime.json','vllm-runtime.json'):
        if read(root/name)['profile'] != config['contract']['runtime_profile']:
            raise ValueError('Actual backend runtime profile differs')
    result = bind_inputs(config, menu, anchors, read(root/'hf.json'), read(root/'vllm.json'))
    if result != read(root/'parity.json') or not result['exact'] or not result['finite_log_policy_supported']:
        raise ValueError('Recomputed exact finite parity required')
    return dict(schema='context-attribute-parity-independent-npu-v1',
        status='finite_six_request_parity_audited',container_id=launch['container_id'],
        source_manifest_sha256=receipt['source_manifest_sha256'],policy_id=REFERENCE_ID,
        cpu_evidence_sha256=sha(cpu_evidence),collected_files=files,parity=result,
        scope='Six TRAIN requests with CPU tensor identity binding; not full population, distributed update, restore, training admission or efficacy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('prepared','launch','inspect','collection','transport','cpu-evidence','cpu-collection','output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = audit(args.prepared,read(args.launch),read(args.inspect),args.collection,
                   read(args.transport),args.cpu_evidence,args.cpu_collection)
    with Path(args.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],parity=result['parity'])))
