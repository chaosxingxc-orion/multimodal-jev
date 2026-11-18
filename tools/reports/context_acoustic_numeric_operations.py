"""Seal and launch a label-free convolution diagnostic using the shared lifecycle."""
import argparse
import json
from pathlib import Path
import re
import shutil

from multimodal_jev.local_config import load_config
from tools.reports.prepare_context_batch_probe_candidate import sha, IMAGE, ROOT, command
from tools.reports.prepare_context_dual_probe_candidate import rebind_argv
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.context_dual_probe_operations import check_runtime

PARENT_SHA = 'eeff28dc8c7f408ad511a1fe72ee50ad1497faad4f4c3df301441cf9a83af2bb'
FILES = ['src/multimodal_jev/__init__.py',
         'src/multimodal_jev/reward.py', 'src/multimodal_jev/task_schema.py',
         'src/multimodal_jev/answers.py', 'src/multimodal_jev/thinking.py',
         'src/multimodal_jev/span_fixed_shape_linear.py',
         'src/multimodal_jev/local_runtime_storage.py', 'src/multimodal_jev/local_checkpoint.py',
         'src/multimodal_jev/span_request_identity.py', 'tools/reports/run_context_batch_probe_candidate.py',
         'tools/reports/context_acoustic_numeric_probe.py']


def argv(c):
    args = rebind_argv(command(c['model']), ROOT, c['remote_root'], c['model'], c['model'],
                       c['driver_root'], 7, c['tag'])
    old = c['remote_root']+'/package/tools/reports/run_context_dual_probe_candidate.py'
    args[args.index(old)] = c['remote_root']+'/package/tools/reports/context_acoustic_numeric_probe.py'
    idx = args.index('--entrypoint')
    args[idx:idx] = ['-v', c['cache_root']+':'+c['cache_root']+':ro']
    return args


def bindings(c):
    for k in ('remote_root', 'cache_root', 'model', 'driver_root'):
        value = c[k]
        if not re.fullmatch('/[a-zA-Z0-9_./-]+', value) or '..' in Path(value).parts:
            raise ValueError('explicit safe path required: '+k)
    if not re.fullmatch('[a-z0-9-]+', c['tag']) or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*', c['host']):
        raise ValueError('safe host/tag required')
    root = Path(c['remote_root'])
    for key in ('cache_root', 'model', 'driver_root'):
        other = Path(c[key])
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise ValueError('runtime must be disjoint from read-only assets')
    if c['priority_mode'] != 'idle_resource_window':
        raise ValueError('idle-resource diagnostic only')


def prepare(c, provenance):
    bindings(c)
    source = Path(c['source']); dest = Path(c['prepared']); package = dest/'package'
    receipts = Path(c['cache_receipts'])
    if sha(receipts) != c['cache_receipts_sha256']:
        raise ValueError('input receipts changed')
    rows = json.loads(receipts.read_text())
    selected = []
    for length in c['padded_frames']:
        candidates = [r for r in rows if r['tensors']['input_features']['shape'] == [1, 128, length]]
        if not candidates or length < 100 or length > 3200 or length % 100:
            raise ValueError('bounded real input at declared length required')
        selected.append(sorted(candidates, key=lambda r: r['id'])[0])
    dest.mkdir(parents=True, exist_ok=False); package.mkdir()
    for name in FILES:
        target = package/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    runtime = dict(model=c['model'], cache_root=c['cache_root'], rows=selected,
                   selection='first lexical ID per configured padded frame length; no labels or predictions',
                   input_receipts_sha256=sha(receipts))
    write_new(package/'config.json', runtime)
    write_new(package/'FILES.json', {str(p.relative_to(package)): sha(p) for p in package.rglob('*') if p.is_file()})
    receipt = dict(status='sealed_numeric_diagnostic', image=IMAGE, argv=argv(c),
                   files_sha256=verify(package), provenance=provenance, training_dispatch_allowed=False)
    write_new(dest/'PREPARED.json', receipt)
    write_new(dest/'operations.json', c)
    return validate(c)[0]


def validate(c):
    bindings(c)
    base = Path(c['prepared']); receipt = json.loads((base/'PREPARED.json').read_text())
    runtime = json.loads((base/'package/config.json').read_text())
    if (receipt['image'] != IMAGE or receipt['argv'] != argv(c)
            or receipt['files_sha256'] != verify(base/'package')
            or receipt['status'] != 'sealed_numeric_diagnostic'
            or receipt['training_dispatch_allowed'] is not False
            or runtime['model'] != c['model'] or runtime['cache_root'] != c['cache_root']
            or runtime['input_receipts_sha256'] != c['cache_receipts_sha256']):
        raise ValueError('sealed diagnostic identity differs')
    for name in FILES:
        if sha(base/'package'/name) != sha(Path(c['source'])/name):
            raise ValueError('diagnostic source changed after sealing')
    return receipt, runtime


def audit(c, collection, handle):
    receipt, runtime = validate(c)
    root = Path(collection); launch = json.loads(Path(handle).read_text())
    inspect = json.loads((Path(c['operations_root'])/(c['tag']+'-inspect.json')).read_text())
    check_runtime(receipt, launch, inspect)
    transport = json.loads(root.with_suffix('.transport.json').read_text())
    if (transport['status'] != 'transport_complete_independent_audit_required'
            or transport['container_id'] != launch['container_id']
            or transport['archive_sha256'] != sha(root.with_suffix('.tar'))
            or transport['files'] != {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
            or sha(root/'config.json') != sha(Path(c['prepared'])/'package/config.json')):
        raise ValueError('collected diagnostic binding differs')
    report = json.loads((root/'evaluation/report.json').read_text())
    terminal = json.loads((root/'evaluation/terminal.json').read_text())
    source = json.loads((root/'evaluation/source.json').read_text())
    if (terminal != {'exit_code': 0} or source['files_sha256'] != receipt['files_sha256']
            or [r['id'] for r in report['rows']] != [r['id'] for r in runtime['rows']]
            or report['full_model_parity_verified'] is not False):
        raise ValueError('incomplete diagnostic')
    return dict(status='numeric_diagnostic_collected_not_full_model_admission',
                files_sha256=receipt['files_sha256'], report_sha256=sha(root/'evaluation/report.json'),
                rows=len(report['rows']), training_dispatch_allowed=False)


def namespace():
    path = Path(__file__).with_name('context_dual_probe_operations.py')
    if sha(path) != PARENT_SHA:
        raise ValueError('shared finite lifecycle changed')
    ns = {'__name__': __name__+'.lifecycle'}
    exec(compile(path.read_text(), str(path), 'exec'), ns)
    ns.update(validate=validate, audit=audit)
    return ns


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    p.add_argument('--action', choices=['prepare', 'validate', 'launch'], required=True)
    a = p.parse_args(); c, provenance = load_config(a.config, a.local)
    result = prepare(c, provenance) if a.action == 'prepare' else validate(c)[0] if a.action == 'validate' else namespace()['launch'](c, provenance)
    print(json.dumps(result, indent=2))
