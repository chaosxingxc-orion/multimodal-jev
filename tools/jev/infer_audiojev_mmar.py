#!/usr/bin/env python3
"""Original-order MMAR inference through the pinned, unmodified CUDA backend."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.mmar import native_prompt, selected_index

OFFICIAL_COMMIT = '8eeeab5e77d2c410fd6058e7bdf2b4125f11903e'


def run(config, output):
    repo = Path(config['official_source']).resolve()
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    if head != OFFICIAL_COMMIT or dirty:
        raise ValueError('official source must be clean and pinned')
    sys.path.insert(0, str(repo))
    from audiojev_inference.api import AudioInput, render_prompt
    from audiojev_inference.backend import OmniBackend
    import audiojev_inference.backend as backend_module
    if Path(backend_module.__file__).resolve().parent.parent != repo:
        raise ValueError('wrong imported backend')
    import torch
    if not config['device'].startswith('cuda:') or not torch.cuda.is_available():
        raise RuntimeError('official baseline requires allocated CUDA; no CPU/NPU fallback')
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'transformers', 'peft', 'numpy', 'scipy', 'soundfile', 'accelerate')}
    if versions['transformers'] != '4.57.6' or not versions['torch'].startswith('2.8.'):
        raise ValueError('official torch/Transformers versions required')
    if not versions['peft'].startswith('0.17.'):
        raise ValueError('official PEFT version required')
    manifest = Path(config['inputs']); checkpoint = Path(config['checkpoint'])
    release = json.loads((manifest.parent / 'MANIFEST.json').read_text())
    if file_hash(manifest) != release['inputs_sha256']:
        raise ValueError('release inputs changed')
    weights = {str(p.relative_to(checkpoint)): file_hash(p) for p in sorted(checkpoint.rglob('*'))
               if p.is_file() and p.suffix in {'.json', '.jinja', '.safetensors', '.model', '.txt'}}
    if not any(k.endswith('.safetensors') for k in weights):
        raise ValueError('missing checkpoint weights')
    identity = {'config': config, 'official_commit': head, 'versions': versions,
                'release_id': release['release_id'],
                'checkpoint_files': weights, 'inputs_sha256': file_hash(manifest),
                'runner_sha256': file_hash(__file__),
                'input_code_sha256': file_hash(Path(__file__).parents[2] / 'src/multimodal_jev/mmar.py'),
                'official_code_sha256': {str(p.relative_to(repo)): file_hash(p)
                                        for p in sorted((repo / 'audiojev_inference').glob('*.py'))}}
    if (output / 'identity.json').exists() and json.loads((output / 'identity.json').read_text()) != identity:
        raise ValueError('resume identity mismatch')
    atomic_json(output / 'identity.json', identity)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != 1000 or len({r['id'] for r in rows}) != 1000:
        raise ValueError('full 1000-question release required')
    if any(set(r) != {'id', 'input', 'audio'} or set(r['input']) != {'question', 'choices', 'audio'} for r in rows):
        raise ValueError('unexpected input fields; supervision must be separate')
    atomic_json(output / 'progress.json', {'phase': 'loading', 'heartbeat_unix': time.time()})
    backend = OmniBackend(checkpoint, device=config['device'], local_files_only=True,
                          max_prompt_tokens=config['max_prompt_tokens'])
    cache = output / 'records'; cache.mkdir(exist_ok=True); records = []
    for index, row in enumerate(rows):
        key = row['id']; inp = row['input']
        if file_hash(inp['audio']) != row['audio']['sha256']:
            raise ValueError('audio changed')
        # Official feature extractor truncates after its maximum. Fail instead
        # of accidentally scoring only part of a publisher waveform.
        if row['audio']['frames'] / row['audio']['sample_rate'] > backend.processor.feature_extractor.chunk_length:
            raise ValueError('audio exceeds official extractor limit')
        prompt = native_prompt(inp, render_prompt)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        path = cache / (hashlib.sha256(key.encode()).hexdigest() + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            if record['id'] != key or record['prompt_sha256'] != prompt_hash:
                raise ValueError('cached prompt mismatch')
            if record['selected_index'] != selected_index(record['probabilities'], len(inp['choices'])):
                raise ValueError('invalid cached selection')
        else:
            start = time.monotonic()
            predictions, usage = backend.predict_many(
                AudioInput(path=inp['audio']), [(key, prompt, len(inp['choices']))])
            probabilities = predictions[key]
            record = {'id': key, 'probabilities': probabilities,
                      'selected_index': selected_index(probabilities, len(inp['choices'])),
                      'prompt_sha256': prompt_hash, 'usage': usage,
                      'seconds': time.monotonic() - start}
            atomic_json(path, record)
        records.append(record)
        atomic_json(output / 'progress.json', {'done': index + 1, 'total': len(rows),
                                               'heartbeat_unix': time.time()})
    predictions = output / 'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(r) + '\n' for r in records))
    atomic_json(output / 'result.json', {'status': 'passed', 'records': len(records),
                'predictions_sha256': file_hash(predictions),
                'scope': 'single released AudioJev checkpoint; official backend; original candidate order; not three-seed training reproduction'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'result.json').exists():
            raise FileExistsError('completed run')
        try:
            run(json.loads(args.config.read_text()), args.output)
        except BaseException as exc:
            atomic_json(args.output / 'terminal.json', {'exit_code': 1, 'error': repr(exc), 'finished_unix': time.time()})
            raise
        atomic_json(args.output / 'terminal.json', {'exit_code': 0, 'finished_unix': time.time()})
