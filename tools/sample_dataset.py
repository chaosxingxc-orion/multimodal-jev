"""Deterministic, without-replacement JSONL sampling from declared TRAIN data."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from multimodal_jev.local_config import load_config
from tools.reports.committed_signal_report import check_paths, read, sha, verify_sources, write


def select(config):
    required = {'schema', 'input', 'input_sha256', 'output_directory', 'seed', 'count',
                'id_field', 'split_field', 'train_split', 'class_field', 'strategy'}
    if set(config) != required or config['schema'] != 'training-dataset-sample-v1':
        raise ValueError('Unsupported sampling configuration')
    if (type(config['seed']) is not int or config['seed'] < 0
            or type(config['count']) is not int or config['count'] < 1
            or config['strategy'] not in ('uniform', 'balanced')
            or config['train_split'] not in ('train', 'learning')
            or any(not isinstance(config[k], str) or not config[k]
                   for k in ('id_field', 'split_field', 'class_field'))):
        raise ValueError('Explicit TRAIN sampling fields, seed and positive count required')
    path = Path(config['input'])
    if sha(path) != config['input_sha256']:
        raise ValueError('Input identity changed')
    rows, seen, classes = [], set(), {}
    with path.open('rb') as stream:
        for line in stream:
            row = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            if not isinstance(row, dict):
                raise ValueError('JSON object rows required')
            uid, label = row.get(config['id_field']), row.get(config['class_field'])
            if (not isinstance(uid, str) or not uid or uid in seen
                    or not isinstance(label, str) or not label
                    or row.get(config['split_field']) != config['train_split']):
                raise ValueError('Distinct IDs, class labels and TRAIN-only rows required')
            seen.add(uid)
            key = hashlib.sha256(json.dumps([config['seed'], uid], ensure_ascii=False).encode()).hexdigest()
            item = (key, uid, label, line)
            rows.append(item)
            classes.setdefault(label, []).append(item)
    if config['count'] > len(rows):
        raise ValueError('Sample exceeds population; replacement forbidden')
    if config['strategy'] == 'uniform':
        chosen = sorted(rows)[:config['count']]
    else:
        # Round-robin ranked classes: exhaust rare classes, never duplicate them.
        groups = [sorted(classes[label]) for label in sorted(classes, key=lambda label:
            hashlib.sha256(json.dumps([config['seed'], label]).encode()).hexdigest())]
        chosen, depth = [], 0
        while len(chosen) < config['count']:
            for group in groups:
                if depth < len(group) and len(chosen) < config['count']:
                    chosen.append(group[depth])
            depth += 1
    if sha(path) != config['input_sha256']:
        raise ValueError('Input changed during sampling')
    counts = {label: sum(item[2] == label for item in chosen) for label in sorted(classes)}
    return chosen, dict(population=len(rows), sampled=len(chosen), seed=config['seed'],
        strategy=config['strategy'], selected_ids=[item[1] for item in chosen],
        population_by_class={k: len(v) for k, v in sorted(classes.items())}, sampled_by_class=counts,
        replacement=False, split=config['train_split'], training_dispatched=False,
        scope='Declared row split enforced; original dataset split provenance must be audited upstream')


def run(template, local):
    config, provenance = load_config(template, local)
    chosen, summary = select(config)
    output = Path(config['output_directory']).resolve()
    sources = {str(Path(p).resolve()): sha(p) for p in
               (__file__, 'src/multimodal_jev/local_config.py',
                'src/multimodal_jev/local_runtime_storage.py',
                'tools/reports/committed_signal_report.py')}
    sources.update({str(Path(template).resolve()): provenance['template_sha256'],
                    str(Path(local).resolve()): provenance['local_bindings_sha256'],
                    str(Path(config['input']).resolve()): config['input_sha256']})
    if any(Path(p).is_relative_to(output) for p in sources):
        raise ValueError('Output must be separate from inputs')
    verify_sources(sources)
    check_paths({'sample': output})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = output / 'COMPLETE.json'
        if complete.exists():
            previous = read(complete)
            if previous['provenance'] != provenance:
                raise ValueError('Configuration changed; use a new output version')
            verify_sources(previous['sources'])
            verify_sources({str(output / p): h for p, h in previous['outputs'].items()})
            return dict(status='already_complete', **previous['summary'])
        if (output / 'status.json').exists():
            raise ValueError('Partial sampling preserved; no automatic retry')
        write(output / 'status.json', dict(phase='sampling'))
        try:
            with (output / 'sample.jsonl').open('xb') as stream:
                for _, _, _, line in chosen:
                    stream.write(line if line.endswith(b'\n') else line + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
            write(output / 'sample-manifest.json', summary)
            verify_sources(sources)
            write(complete, dict(schema='training-dataset-sample-publication-v1',
                provenance=provenance, sources=sources, summary=summary,
                outputs={p: sha(output / p) for p in ('sample.jsonl', 'sample-manifest.json')}))
            write(output / 'status.json', dict(phase='complete'))
            return dict(status='complete', **summary)
        except Exception as exc:
            write(output / 'status.json', dict(phase='failed_preserve_evidence', error=repr(exc)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', default=os.environ.get('JEV_LOCAL_CONFIG', '.local/config/macros.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.local)))


if __name__ == '__main__':
    main()
