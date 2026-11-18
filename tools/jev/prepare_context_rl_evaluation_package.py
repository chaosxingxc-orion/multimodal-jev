#!/usr/bin/env python3
"""Package one completed formal RL checkpoint for isolated fixed-image scoring."""
import argparse
import json
from pathlib import Path
import re
import shutil

if __package__:
    from .run_context_rl_selection import prepare
    from .context_rl_evaluation_plan import sha
else:
    from run_context_rl_selection import prepare
    from context_rl_evaluation_plan import sha

IMAGE = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
PREFIX = '/data/xingchao/multimodal-jev/simple-final-v1/'
TOOLS = ['run_context_rl_selection.py', 'context_rl_evaluation_plan.py',
         'probe_context_engine_audio.py', 'evaluate_context_selection_vllm.py',
         'evaluate_context_event_records.py']


def verify(package, digest):
    root = Path(package)
    if sha(root / 'PACKAGE.json') != digest:
        raise ValueError('evaluation package manifest changed')
    m = json.loads((root / 'PACKAGE.json').read_text())
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink():raise ValueError('package symlink forbidden')
        if p.is_file() and p != root / 'PACKAGE.json':actual.add(str(p.relative_to(root)))
    if actual != set(m['files']):raise ValueError('package file population differs')
    for name, expected in m['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(root / name) != expected:
            raise ValueError('package file changed: ' + name)
    if m['image'] != IMAGE or not re.fullmatch(re.escape(PREFIX) + r'context-rl-eval-[a-z0-9-]+', m['remote_root']):
        raise ValueError('fixed image and isolated evaluation root required')
    return m


def package(*, specification, output, remote_root):
    if not re.fullmatch(re.escape(PREFIX) + r'context-rl-eval-[a-z0-9-]+', remote_root):
        raise ValueError('new context-rl-eval name under fixed local storage required')
    spec_path = Path(specification)
    spec = json.loads(spec_path.read_text())
    args = dict(spec)
    for name in ('template', 'experiment', 'stage1_proof', 'rl_proof'):
        args[name] = json.loads(Path(spec[name]).read_text())
    # Reject live/prefix/missing full sources before creating any package.
    c = prepare(**args)
    output = Path(output)
    output.mkdir(exist_ok=False)
    source = Path(c['source_policy'])
    shutil.copytree(source, output / 'policy')
    c['source_policy'] = remote_root + '/policy'
    c['physical_devices'] = [7]
    copied = {}
    paths = sorted(Path('src').rglob('*.py')) + [Path('tools/jev') / n for n in TOOLS]
    for path in paths:
        target = output / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied[str(path)] = sha(path)
    # These are the only external pinned dependencies read by the scoring
    # entry; stale Stage1 code/diagnostic branch paths are not carried forward.
    external = {p: h for p, h in c['pins'].items()
                if Path(p).is_relative_to(Path(c['model'])) or p == c['tasks']}
    if str(Path(c['model']) / 'model.safetensors') not in external or c['tasks'] not in external:
        raise ValueError('pinned model weights and task definitions required')
    if not all(p.startswith('/mnt/cephfs/xingchao/') for p in [c['model'], c['tasks'], c['release']]):
        raise ValueError('shared read-only model/data mounts required')
    for path, expected in external.items():
        if sha(path) != expected:raise ValueError('external dependency changed: ' + path)
    # One dedicated existing cache; no accidental writes back to collected inputs.
    cache = PREFIX + 'context-selection-input-cache-v1'
    if c['encoded_audio_cache'] != cache:raise ValueError('fixed selection input cache required')
    c['pins'] = external
    originals = dict(c['rl_evaluation']['binding_files'])
    originals[str(spec_path.resolve())] = sha(spec_path)
    for name in ('template', 'experiment', 'stage1_proof', 'rl_proof'):
        originals[str(Path(spec[name]).resolve())] = sha(spec[name])
    bindings = {}
    (output / 'evidence').mkdir()
    for i, (path, digest) in enumerate(sorted(originals.items())):
        if sha(path) != digest:raise ValueError('preparation source changed')
        relative = f'evidence/{i:03d}.json'
        shutil.copyfile(path, output / relative)
        bindings[remote_root + '/' + relative] = digest
    c['rl_evaluation']['binding_files'] = bindings
    for unused in ('processor', 'reference_branch'):
        c.pop(unused, None)
    for path in output.rglob('*'):
        if path.is_file():c['pins'][remote_root + '/' + str(path.relative_to(output))] = sha(path)
    (output / 'config.json').write_text(json.dumps(c, indent=2) + '\n')
    manifest = dict(status='prepared_only_no_inference', image=IMAGE, remote_root=remote_root,
        step=c['checkpoint_step'], run_id=c['rl_evaluation']['run_id'],
        algorithm=c['rl_evaluation']['algorithm'], source_policy_id=c['source_policy_id'],
        preparation_inputs=originals, source_files=copied,
        files={str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()})
    (output / 'PACKAGE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(output, sha(output / 'PACKAGE.json'))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remote-root', required=True)
    a = p.parse_args()
    m = package(specification=a.spec, output=a.output, remote_root=a.remote_root)
    print(json.dumps(dict(files=len(m['files']), status=m['status'],
        manifest_sha256=sha(a.output / 'PACKAGE.json')), indent=2))
