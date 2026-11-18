"""Prepare a label-free FSC pair for the fixed shared inference image.

No remote calls. Source policy selection must already be independently sealed.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

from tools.jev.prepare_fsc_context_source import verify as verify_source, sha
from tools.jev.prepare_context_rl_evaluation_package import IMAGE, PREFIX
from infer_fsc_context_transfer import load_inputs

CACHE = PREFIX+'fsc-context-processor-cpu-v1/run/cache'
MODEL = '/mnt/cephfs/xingchao/models/Qwen__Qwen3-ASR-0.6B-hf'


def read(path):
    return json.loads(Path(path).read_text())


def verify(package, digest):
    root = Path(package)
    if root.is_symlink() or sha(root/'PACKAGE.json') != digest:
        raise ValueError('transfer runtime manifest changed')
    m = read(root/'PACKAGE.json')
    if (m['image'] != IMAGE or m['status'] != 'prepared_fsc_pair_no_dispatch'
            or not re.fullmatch(re.escape(PREFIX)+r'fsc-context-pair-[a-z0-9-]+', m['remote_root'])):
        raise ValueError('fixed image and isolated transfer directory required')
    actual = {}
    for p in root.rglob('*'):
        if p.is_symlink():raise ValueError('runtime package symlink forbidden')
        if p.is_file() and p != root/'PACKAGE.json':actual[str(p.relative_to(root))] = sha(p)
    if actual != m['files']:raise ValueError('runtime package files differ')
    source = verify_source(root/'frozen-source', m['source_manifest_sha256'])
    config = read(root/'config.json');remote = m['remote_root']
    expected_inputs = dict(input_manifest=dict(path=remote+'/inputs/MANIFEST.json',
                           sha256=sha(root/'inputs/MANIFEST.json')),
                           input_audit=dict(path=remote+'/inputs/audit.json', sha256=sha(root/'inputs/audit.json')),
                           requests=remote+'/inputs/requests.jsonl', input_receipts=remote+'/inputs/receipts.jsonl',
                           cache_root=CACHE)
    if (config['inputs'] != expected_inputs or config['model'] != MODEL or config['physical_devices'] != [7]
            or config['source_package'] != remote+'/frozen-source'
            or config['source_manifest_sha256'] != m['source_manifest_sha256']
            or sha(root/'inputs/audit.json') != source['input_audit_sha256']):
        raise ValueError('runtime source/input/device binding differs')
    local_inputs = dict(expected_inputs, requests=str(root/'inputs/requests.jsonl'),
                        input_receipts=str(root/'inputs/receipts.jsonl'))
    local_inputs['input_manifest'] = dict(expected_inputs['input_manifest'],path=str(root/'inputs/MANIFEST.json'))
    local_inputs['input_audit'] = dict(expected_inputs['input_audit'],path=str(root/'inputs/audit.json'))
    load_inputs(local_inputs)
    expected_pins = {remote+'/'+name:h for name,h in actual.items()
                     if name.startswith(('source/', 'frozen-source/', 'inputs/'))}
    expected_pins.update(m['external_pins'])
    if config['pins'] != expected_pins:
        raise ValueError('runtime dependency pins differ')
    for name,digest in m['external_pins'].items():
        if Path(name).parent != Path(MODEL) or sha(name) != digest:
            raise ValueError('pinned base assets differ')
    if MODEL+'/model.safetensors' not in m['external_pins']:
        raise ValueError('pinned base weights required')
    return m


def prepare(spec, output, remote_root):
    if not re.fullmatch(re.escape(PREFIX)+r'fsc-context-pair-[a-z0-9-]+', remote_root):
        raise ValueError('new isolated transfer directory required')
    source = Path(spec['source_package']).resolve()
    m = verify_source(source, spec['source_manifest_sha256'])
    inputs = spec['inputs']
    if set(inputs) != {'input_manifest','input_audit','requests','input_receipts','cache_root'} or inputs['cache_root'] != CACHE:
        raise ValueError('only fixed label-free audited inputs required')
    _, _, proof = load_inputs(inputs)
    if (inputs['input_audit']['sha256'] != m['input_audit_sha256']
            or proof['manifest']['processor_sha256'] != m['processor_sha256']):
        raise ValueError('source/cache processor differs')
    output = Path(output).resolve()
    for path in (source, Path(inputs['requests']).parent, Path(MODEL)):
        if output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output):
            raise ValueError('independent runtime package required')
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'fsc_runtime_package':output})
    external = {str(p):sha(p) for p in Path(MODEL).iterdir()
                if p.is_file() and (p.suffix in ('.json','.txt','.jinja') or p.name == 'model.safetensors')}
    if MODEL+'/model.safetensors' not in external:raise ValueError('base weights required')
    output.mkdir(exist_ok=False)
    shutil.copytree(source, output/'frozen-source')
    (output/'inputs').mkdir()
    for src, name in [(inputs['input_manifest']['path'],'MANIFEST.json'),
                      (inputs['input_audit']['path'],'audit.json'),
                      (inputs['requests'],'requests.jsonl'), (inputs['input_receipts'],'receipts.jsonl')]:
        shutil.copyfile(src, output/'inputs'/name)
    # Copy the shared Python dependency tree; do not clone a new training stack.
    for folder in ('src', 'tools/jev'):
        for path in sorted(Path(folder).rglob('*.py')):
            target = output/'source'/path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    relocated = dict(input_manifest=dict(path=remote_root+'/inputs/MANIFEST.json',
                      sha256=sha(output/'inputs/MANIFEST.json')),
                      input_audit=dict(path=remote_root+'/inputs/audit.json',sha256=sha(output/'inputs/audit.json')),
                      requests=remote_root+'/inputs/requests.jsonl',input_receipts=remote_root+'/inputs/receipts.jsonl',
                      cache_root=CACHE)
    config = dict(source_package=remote_root+'/frozen-source', source_manifest_sha256=spec['source_manifest_sha256'],
                  model=MODEL,physical_devices=[7],inputs=relocated,
                  pins={**external,**{remote_root+'/'+str(p.relative_to(output)):sha(p)
                        for p in output.rglob('*') if p.is_file()}})
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest = dict(status='prepared_fsc_pair_no_dispatch',image=IMAGE,remote_root=remote_root,
                    source_manifest_sha256=spec['source_manifest_sha256'],external_pins=external,
                    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'PACKAGE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return verify(output,sha(output/'PACKAGE.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remote-root',required=True)
    a = p.parse_args()
    result = prepare(read(a.spec),a.output,a.remote_root)
    print(json.dumps(dict(status=result['status'],files=len(result['files']),
                         manifest_sha256=sha(a.output/'PACKAGE.json')),indent=2))
