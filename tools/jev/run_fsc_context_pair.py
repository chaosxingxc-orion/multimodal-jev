"""Run source-selected FSC policies in one externally leased container.

Both 31-candidate parity panels must pass before either full inference starts.
No device allocation, target labels, scoring, retry or target checkpoint choice.
The launcher/independent collector remain separate responsibilities.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.jev.prepare_fsc_context_source import verify, sha
from infer_fsc_context_transfer import load_inputs, validate_source
from probe_fsc_context_parity import compare, panel_indices


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def configs(spec):
    package = Path(spec['source_package']).resolve()
    manifest = verify(package, spec['source_manifest_sha256'])
    shared = spec['inputs']
    if set(shared) != {'input_manifest', 'input_audit', 'requests', 'input_receipts', 'cache_root'}:
        raise ValueError('only audited label-free input configuration allowed')
    rows, receipts, proof = load_inputs(shared)
    if (shared['input_audit']['sha256'] != manifest['input_audit_sha256']
            or proof['manifest']['processor_sha256'] != manifest['processor_sha256']):
        raise ValueError('source pair and input audit differ')
    device = spec['physical_devices']
    if len(device) != 1 or type(device[0]) is not int or device[0] not in range(8):
        raise ValueError('one physical NPU required')
    # The future package must bind all Python sources consumed by the children.
    pins = spec['pins']
    sources = [p.resolve() for directory in (Path(__file__).parent, Path(__file__).parents[2]/'src')
               for p in directory.rglob('*.py')]
    for path in sources:
        if pins.get(str(path)) != sha(path):
            raise ValueError('runtime source is not pinned: ' + str(path))
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('runtime pin changed: ' + path)
    output = {}
    for role in ('initial', 'selected'):
        entry = manifest['policies'][role]
        c = dict(shared, model=spec['model'], base_id=entry['contract']['base_id'],
                 snapshot=str(package/entry['policy']), adapter=str(package/entry['adapter']),
                 policy_id=entry['policy_id'], contract=entry['contract'], source_role=role,
                 source_selection=dict(path=str(package/'source-selection.json'),
                                       sha256=sha(package/'source-selection.json')),
                 physical_devices=device, pins=pins)
        validate_source(c)
        output[role] = c
    return output, rows, receipts


def invoke(script, arguments, log):
    with log.open('x') as stream:
        subprocess.run([sys.executable, str(Path(__file__).parent/script), *map(str, arguments)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def check_panel(directory, config, rows, receipts):
    hf, vl = (read(directory/(phase+'.json')) for phase in ('hf', 'vllm'))
    expected = panel_indices(rows, receipts)
    for result in (hf, vl):
        if (result['policy_id'] != config['policy_id'] or result['indices'] != expected
                or result['input_manifest_sha256'] != config['input_manifest']['sha256']
                or len(result['records']) != len(expected)):
            raise ValueError('parity panel differs from preselected source and inputs')
        for index, record in zip(expected, result['records'], strict=True):
            receipt = receipts[index]
            identity = dict(policy_id=config['policy_id'], positions=receipt['positions'][0],
                            inputs={k:v for k,v in receipt['tensors'].items() if k != 'positions'})
            if record['id'] != rows[index]['id'] or record['request_identity'] != identity:
                raise ValueError('parity panel does not match audited inputs')
    result = compare(hf, vl)
    if not result['exact'] or result != read(directory/'comparison.json'):
        raise ValueError('both raw backends must pass exact probability parity')
    return result


def check_full(directory, config, rows, receipts):
    from infer_fsc_context_transfer import checked_prediction
    predictions = read(directory/'predictions.json')['records']
    result = read(directory/'result.json')
    if (len(predictions) != len(rows) or result['recordings'] != 3793 or result['decisions'] != 31
            or result['policy_id'] != config['policy_id'] or result['source_role'] != config['source_role']
            or result['predictions_sha256'] != sha(directory/'predictions.json')):
        raise ValueError('complete bound inference result required')
    for row, receipt, prediction in zip(rows, receipts, predictions, strict=True):
        if prediction['id'] != row['id']:
            raise ValueError('complete ordered input population required')
        checked_prediction(row, receipt, prediction, config['policy_id'])
    return result


def run(spec, output):
    pair, rows, receipts = configs(spec)
    output = Path(output).resolve()
    for path in [spec['source_package'], spec['model'], spec['inputs']['cache_root'],
                 Path(spec['inputs']['requests']).parent]:
        root = Path(path).resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError('independent output directory required')
    from multimodal_jev.local_runtime_storage import require_training_storage
    require_training_storage(output)
    output.mkdir(exist_ok=False)
    write(output/'input.json', spec)
    last_phase = 'prepared'
    try:
        for role, config in pair.items():
            folder = output/role
            folder.mkdir()
            write(folder/'config.json', config)
        panels = {}
        for role, config in pair.items():
            folder = output/role
            for phase in ('hf', 'vllm'):
                last_phase = role+'-parity-'+phase
                invoke('probe_fsc_context_parity.py', ['--config', folder/'config.json',
                       '--output', folder/'parity', '--phase', phase], folder/(phase+'.log'))
            panels[role] = check_panel(folder/'parity', config, rows, receipts)
        results = {}
        for role, config in pair.items():
            last_phase = role+'-full-inference'
            folder = output/role
            invoke('infer_fsc_context_transfer.py', ['--config', folder/'config.json',
                   '--output', folder/'inference'], folder/'inference.log')
            results[role] = check_full(folder/'inference', config, rows, receipts)
        report = dict(status='pair_inference_complete_independent_collection_pending',
                      source_manifest_sha256=spec['source_manifest_sha256'],
                      panels=panels, policies=results, target_tuning=False,
                      scope='No target scores or independent runtime provenance certification.')
        write(output/'result.json', report)
        write(output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
        return report
    except BaseException as exc:
        write(output/'terminal.json', dict(exit_code=1, failed_phase=last_phase,
              error=repr(exc), retry=False, finished_unix=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(read(args.spec), args.output)
