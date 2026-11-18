"""One CPU process proving release, objective and evaluator compose by name.

Builds a small synthetic official-clean release in the output root (never reads
Ceph, never starts a container), authenticates it through the registered release
component, scores it with the registered objective and evaluates the same rows
with the registered evaluator. The scores are a seeded synthetic matrix, not a
model output: this smoke exercises the seams and their determinism, and says
nothing about model quality.

The acceptance verdict is deterministic and computed here. A non-empty list of
failures is a rejected report, not a warning.
"""
import argparse
import ast
from dataclasses import asdict
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from multimodal_jev import context_component_registry as components
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.context_candidate_task import bind_candidate_targets, compile_candidates
from multimodal_jev.context_evaluation_contract import (
    IDENTITY_FIELDS, METRIC_FIELDS, SCOPE, from_result, ranked_key, validate_row)
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.task_schema import Candidate, DecisionSchema

SCHEMA = 'context-component-smoke-v1'
REQUIRED_KINDS = ('release', 'objective', 'evaluator', 'topology')
# Constructs that would mean the composed path rewrites source at run time.
REWRITE_ATTRIBUTES = ('getsource', 'getsourcelines', 'FunctionType')


def seconds(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def rewriting_sites(path):
    """Line addresses of run-time source rewriting in one module, if any."""
    tree = ast.parse(Path(path).read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in ('exec', 'eval'):
            found.append(f'{Path(path).name}:{node.lineno}:{func.id}')
        elif isinstance(func, ast.Attribute) and func.attr in REWRITE_ATTRIBUTES:
            found.append(f'{Path(path).name}:{node.lineno}:{func.attr}')
    return found


def synthetic_source(root, rows_per_role):
    """A tiny official-clean source bundle shaped like the real publication."""
    import soundfile as sf
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    source = root/'source'
    source.mkdir()
    reviews = {}
    for j, role in enumerate(('learning', 'selection')):
        requests, targets, packet = [], [], []
        for i in range(rows_per_role):
            audio = root/f'{role}-{i}.wav'
            sf.write(audio, np.sin(np.arange(1600) * (.01 + .003 * (j*rows_per_role + i))
                                   ).astype('float32') * .2, 16000)
            request = dict(id='slurp:'+audio.name, official_split='train', role=role,
                           source_group=f'{role}-group-{i}', audio_sha256=file_hash(audio),
                           input={'audio': str(audio)})
            target = dict(id=request['id'], request_sha256=content_hash(request),
                          annotation_status='available',
                          native_target=dict(scenario='alarm', action='set', entities=[]),
                          semantic_review_status='pending', optimization_allowed=False)
            requests.append(request)
            targets.append(target)
            packet.append(dict(request=request, target=target))
        for suffix, data in (('requests', requests), ('targets', targets)):
            (source/f'{role}.{suffix}.jsonl').write_bytes(b''.join(map(canonical, data)))
        packet_path = root/f'{role}-packet.json'
        packet_path.write_bytes(canonical(packet))
        # One row per role stays pending so the admitted population is masked,
        # exactly as the real publication masks known-pending sources.
        if role == 'learning':
            audit = dict(reviewed={str(i): 'fixture' for i in range(rows_per_role - 1)},
                         pending={str(rows_per_role - 1): 'ambiguous fixture'})
        else:
            audit = dict(records=[dict(index=i, id=item['request']['id'],
                                       source_group=item['request']['source_group'],
                                       record_sha256=content_hash(item),
                                       status='pending' if i == rows_per_role - 1
                                       else 'reference_consistent')
                                  for i, item in enumerate(packet)])
        audit['packet_sha256'] = file_hash(packet_path)
        audit_path = root/f'{role}-audit.json'
        audit_path.write_bytes(canonical(audit))
        reviews[role] = [str(packet_path), str(audit_path)]
    manifest = dict(schema='slurp-clean-source-bundles-v1',
                    counts={'learning': rows_per_role, 'selection': rows_per_role},
                    files={p.name: file_hash(p) for p in source.iterdir()})
    (source/'MANIFEST.json').write_bytes(canonical(manifest))
    receipt = root/'receipt.json'
    receipt.write_bytes(canonical(dict(
        release=str(source), manifest_sha256=file_hash(source/'MANIFEST.json'),
        all_native_targets_reconstructed_independently=True,
        cross_role_source_group_or_audio_overlap=0)))
    task = dict(kind='intent',
                schema=asdict(DecisionSchema(
                    'smoke-intent', 'choice', 'Which requested task?',
                    tuple(Candidate(i, i) for i in
                          ('alarm_set', 'calendar_set', 'alarm_query')), False)),
                candidate_pairs={'alarm_set': ['alarm', 'set'],
                                 'calendar_set': ['calendar', 'set'],
                                 'alarm_query': ['alarm', 'query']},
                thinking={'requirements': ['Use only the audio and supplied questions.']})
    tasks = root/'tasks.json'
    tasks.write_bytes(canonical({'intent': task}))
    pins = [source/'MANIFEST.json', receipt, tasks]
    pins += [Path(p) for pair in reviews.values() for p in pair]
    return dict(source=str(source), receipt=str(receipt), tasks=str(tasks), reviews=reviews,
                pins={str(p): file_hash(p) for p in pins})


def build_release(c, out):
    from multimodal_jev.slurp_portable_release import publish
    root = out/'synthetic'
    config = synthetic_source(root, c['rows_per_role'])
    release = out/'release'
    manifest, elapsed = seconds(lambda: publish(config, release))
    digest = file_hash(release/'MANIFEST.json')
    return dict(schema=manifest['schema'], manifest_sha256=digest, root=str(release),
                counts={role: row['recordings'] for role, row in manifest['counts'].items()},
                quarantine=manifest['quarantine'], build_seconds=elapsed)


def load_release(c, release):
    """Authenticate through the registered release handle, twice."""
    digest = release['manifest_sha256']
    handle_class = components.resolve('release', c['release_handle'])
    handle = handle_class(max_entries=c['cache_entries'])
    started = time.perf_counter()
    data, task, manifest, reused_first = handle.load(release['root'], digest, 'learning',
                                                     'optimize')
    again, task2, _, reused_second = handle.load(release['root'], digest, 'learning', 'optimize')
    elapsed = time.perf_counter() - started
    refused = None
    try:
        handle.load(release['root'], digest, 'selection', 'optimize')
    except ValueError as error:
        refused = str(error)
    return dict(rows=len(data), candidates=len(task['schema']['candidates']),
                first_reused=reused_first, second_reused=reused_second,
                identical=again == data and task2 == task,
                selection_optimize_refused=refused, seconds=elapsed,
                handle_class=handle_class.__name__), data, task


def score_objective(c, loss, data, task):
    """Synthetic scores through the registered objective, with a real backward."""
    import torch
    ids = [x['id'] for x in task['schema']['candidates']]
    candidates = [dict(id=x['id'], definition=x['description'])
                  for x in task['schema']['candidates']]
    requirements = list(task['thinking']['requirements'])
    rows = [row for row in data if row['intent'] in ids]
    requests = [compile_candidates(row['audio'], candidates, requirements,
                                   semantics='exclusive_event') for row in rows]
    bound = [bind_candidate_targets(request, semantics='exclusive_event',
                                    exclusive_event=row['intent'])
             for request, row in zip(requests, rows, strict=True)]
    rng = np.random.default_rng(c['seed'])
    logits = torch.tensor(rng.standard_normal((len(rows), len(ids))), dtype=torch.float32,
                          requires_grad=True)
    labels = torch.tensor([item['labels'] for item in bound], dtype=torch.bool)
    observed = torch.tensor([item['supervised'] for item in bound], dtype=torch.bool)
    real = torch.ones(len(rows), dtype=torch.bool)
    weights = real.to(torch.float32)
    value, details = loss(logits, labels, observed, weights, real,
                          global_real_requests=len(rows), world_size=1,
                          temperature=c['temperature'])
    value.backward()
    gradient = logits.grad
    probabilities = torch.sigmoid(logits.detach()).numpy()
    return dict(rows=len(rows), candidates=len(ids), loss=float(value.detach()),
                details=details,
                gradient_finite=bool(torch.isfinite(gradient).all()),
                gradient_norm=float(gradient.norm()),
                one_positive_per_row=int(labels.sum(-1).unique().tolist()[0]),
                scores=probabilities.tolist(), ids=ids), probabilities, ids


def evaluate(c, metrics_class, probabilities, ids, data):
    metrics = metrics_class(ids, threshold=c['threshold'])
    rows = [row for row in data if row['intent'] in ids]
    for row, values in zip(rows, probabilities, strict=True):
        metrics.add(row['id'], row['intent'], [float(v) for v in values])
    result = metrics.result()
    view = from_result(result)
    return dict(metrics=view, selection_key=list(ranked_key(view)),
                identity_keys_present=set(result) == set(METRIC_FIELDS) | set(IDENTITY_FIELDS),
                scope_matches_contract=result['scope'] == SCOPE,
                validated=validate_row(view, METRIC_FIELDS) == view)


def topology(c, derive):
    plan = derive(c['topology'])
    plan['global_batch_expected'] = 64 * c['topology']['training_devices']
    plan['global_batch_matches'] = plan['global_batch'] == plan['global_batch_expected']
    return plan


def components_report(c):
    """Resolved component identities and their files, plus rewrite sites."""
    catalog = {(kind, name): (path, digest) for kind, name, path, digest in components.catalog()}
    rows, sites = [], []
    for kind, name in sorted(c['components'].items()):
        found = components.entry(kind, name)
        path, digest = catalog[(kind, name)]
        rows.append(dict(kind=kind, name=name, module=found['module'],
                         attribute=found['attribute'], status=found['status'],
                         module_file=path, sha256=digest))
        sites.extend(rewriting_sites(path))
    # The handle is selected separately from the release reader it wraps.
    found = components.entry('release', c['release_handle'])
    path, digest = catalog[('release', c['release_handle'])]
    rows.append(dict(kind='release', name=c['release_handle'], module=found['module'],
                     attribute=found['attribute'], status=found['status'],
                     module_file=path, sha256=digest))
    sites.extend(rewriting_sites(path))
    return rows, sorted(set(sites))


def run(c, provenance):
    if c['schema'] != SCHEMA:
        raise ValueError('explicit component smoke schema required')
    if set(c['components']) != set(REQUIRED_KINDS):
        raise ValueError('explicit release/objective/evaluator/topology required')
    if type(c['rows_per_role']) is not int or c['rows_per_role'] < 3:
        raise ValueError('at least three rows per role required so one can stay pending')
    out = Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    commit_json(out/'config.json', c)
    commit_json(out/'provenance.json', provenance)
    failures = []

    # 1. Every name a config may select resolves before any work starts.
    started = time.perf_counter()
    resolved = components.validate(c['components'])
    resolve_seconds = time.perf_counter() - started
    entries, rewrite_sites = components_report(c)
    if rewrite_sites:
        failures.append('source rewriting in the composed path: ' + ', '.join(rewrite_sites))

    # 2. Synthetic release, built and authenticated in this process.
    release, build_seconds = build_release(c, out), None
    build_seconds = release.pop('build_seconds')
    if release['schema'] != 'slurp-official-clean-intent-v2':
        failures.append('release schema is not the portable v2 schema')
    loaded, data, task = load_release(c, release)
    if not loaded['second_reused']:
        failures.append('second authentication of identical content was not reused')
    if loaded['first_reused']:
        failures.append('first authentication reported reuse')
    if not loaded['identical']:
        failures.append('cached authentication returned different evidence')
    if not loaded['selection_optimize_refused']:
        failures.append('selection was not refused as an optimization role')
    if loaded['rows'] != c['rows_per_role'] - 1:
        failures.append('admitted population differs from the pending mask')

    # 3. Objective on synthetic scores, then the same probabilities through the
    #    evaluator. Two passes must be bit-identical.
    scored, probabilities, ids = score_objective(c, resolved['objective'], data, task)
    evaluated = evaluate(c, resolved['evaluator'], probabilities, ids, data)
    repeat = score_objective(c, resolved['objective'], data, task)[0]
    # scores is the reported list projection, so this compares exactly what the
    # report carries rather than a wider in-memory array.
    deterministic = repeat['loss'] == scored['loss'] and repeat['scores'] == scored['scores']
    if not deterministic:
        failures.append('objective is not deterministic on the same inputs')
    if not scored['gradient_finite'] or not scored['gradient_norm'] > 0:
        failures.append('objective gradient is missing or not finite')
    if scored['one_positive_per_row'] != 1:
        failures.append('synthetic supervision is not one positive per row')
    if not evaluated['identity_keys_present'] or not evaluated['scope_matches_contract']:
        failures.append('evaluator result differs from the contract vocabulary')

    plan = topology(c, resolved['topology'])
    if not plan['global_batch_matches']:
        failures.append('topology derivation disagrees with 64 x DP')

    report = dict(
        schema=SCHEMA, status='rejected' if failures else 'accepted',
        model_calls=0, npu_executed=False, failures=failures,
        environment=dict(python=sys.version.split()[0], host=platform.node(),
                         machine=platform.machine(),
                         torch=__import__('torch').__version__),
        implementation_sha256=file_hash(__file__),
        provenance=provenance,
        components=entries, source_rewriting_sites=rewrite_sites,
        resolve_seconds=resolve_seconds, build_seconds=build_seconds,
        release=release, load=loaded, objective=scored, evaluation=evaluated,
        topology=plan, deterministic=deterministic)
    commit_json(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    c, provenance = load_config(args.config, args.local)
    report = run(c, provenance)
    print(json.dumps(dict(status=report['status'], failures=report['failures'],
                          objective=report['objective']['loss'],
                          evaluation=report['evaluation']['metrics']), indent=2))
    if report['status'] != 'accepted':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
