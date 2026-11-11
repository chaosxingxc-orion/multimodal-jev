"""Immutable publication of source-authenticated, reviewed SLURP decisions."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile

from .baseline import file_hash
from .consumption import canonical
from .slurp_training import content_hash, project


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def publish(source, source_receipt, tasks_path, ledger_path, destination):
    source, source_receipt, tasks_path, ledger_path, destination = map(
        Path, (source, source_receipt, tasks_path, ledger_path, destination))
    receipt = json.loads(source_receipt.read_text())
    manifest_hash = file_hash(source / 'MANIFEST.json')
    if str(source.resolve()) != receipt['release'] or manifest_hash != receipt['manifest_sha256']:
        raise ValueError('verified clean source release required')
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    if manifest['schema'] != 'slurp-clean-source-bundles-v1':
        raise ValueError('clean source protocol required')
    for name in ('learning.requests.jsonl', 'learning.targets.jsonl'):
        if file_hash(source / name) != manifest['files'][name]:
            raise ValueError('learning source changed')
    tasks = json.loads(tasks_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    if (ledger['source_manifest_sha256'] != manifest_hash
            or ledger['tasks_sha256'] != file_hash(tasks_path)):
        raise ValueError('review ledger source/task identity mismatch')
    for path, expected in ledger['evidence_files'].items():
        if not Path(path).is_absolute() or file_hash(path) != expected:
            raise ValueError('review evidence changed')
    if not ledger['evidence_files'] or not ledger['scope']:
        raise ValueError('explicit review evidence and scope required')
    entries = ledger['records']
    if not entries or len({r['id'] for r in entries}) != len(entries):
        raise ValueError('nonempty unique review population required')
    wanted = {r['id']:r for r in entries}
    requests = {}
    for request in read_rows(source / 'learning.requests.jsonl'):
        if request['id'] in wanted:
            if request['id'] in requests:
                raise ValueError('duplicate source request')
            requests[request['id']] = request
    if set(requests) != set(wanted):
        raise ValueError('review population must belong to learning')
    targets = {}
    for target in read_rows(source / 'learning.targets.jsonl'):
        if target['id'] in wanted:
            if target['id'] in targets:
                raise ValueError('duplicate source target')
            targets[target['id']] = target
    if set(targets) != set(wanted):
        raise ValueError('review population missing source target')
    identity = dict(source_manifest_sha256=manifest_hash,
                    source_receipt_sha256=file_hash(source_receipt),
                    tasks_sha256=file_hash(tasks_path), ledger_sha256=file_hash(ledger_path),
                    implementation={name:file_hash(Path(__file__).with_name(name)) for name in
                        ('slurp_training_release.py','slurp_training.py','task_schema.py','thinking.py','consumption.py')})
    release_id = content_hash(identity)
    output = destination / release_id
    if output.exists():
        raise FileExistsError('immutable training release exists')
    destination.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.reviewed-', dir=destination))
    counts, labels, groups, checked_audio = Counter(), Counter(), set(), {}
    try:
        with (stage / 'train.jsonl').open('wb') as train, (stage / 'masks.jsonl').open('wb') as masks:
            for uid, entry in sorted(wanted.items()):
                request, target = requests[uid], targets[uid]
                # Do not silently expand a sparse reviewer ledger with negative gold.
                if set(entry['reviews']) != set(tasks):
                    raise ValueError('explicit status for every fixed task required')
                examples, statuses = project(request, target, tasks, entry['reviews'], manifest_hash)
                if examples:
                    audio = request['input']['audio']
                    if audio not in checked_audio:
                        checked_audio[audio] = file_hash(audio)
                    if checked_audio[audio] != request['audio_sha256']:
                        raise ValueError('reviewed audio bytes changed')
                    groups.add(request['source_group'])
                for example in examples:
                    train.write(canonical(example.to_record()))
                    labels[(example.example_id.rsplit('::',1)[1], str(example.target))] += 1
                counts.update(statuses.values())
                masks.write(canonical(dict(id=uid, source_group=request['source_group'], statuses=statuses)))
        if counts['reviewed'] == 0:
            raise ValueError('no reviewed decisions; do not publish an empty training release')
        shutil.copyfile(tasks_path, stage / 'tasks.json')
        shutil.copyfile(ledger_path, stage / 'review-ledger.json')
        for path in stage.iterdir():
            path.chmod(0o600)
        report = dict(schema='slurp-reviewed-training-release-v1', release_id=release_id,
            identity=identity, source_release=str(source.resolve()),
            counts=dict(counts), reviewed_recordings=len(checked_audio), source_groups=len(groups),
            decisions=counts['reviewed'], requested_recordings=len(entries),
            labels=[dict(task=k,value=v,count=n) for (k,v),n in sorted(labels.items())],
            files={p.name:file_hash(p) for p in sorted(stage.iterdir())},
            review_scope=ledger['scope'], selection_opened=False, test_opened=False,
            optimization_scope='Admitted decisions only; no evaluation release, rule-family generalization or model-quality claim.')
        (stage / 'MANIFEST.json').write_bytes(canonical(report))
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return output, report
