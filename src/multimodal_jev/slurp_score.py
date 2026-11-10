"""Complete-recording SLURP scoring using pinned upstream metric implementations."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from .baseline import file_hash
from .consumption import canonical
from .slurp_native import parse_prediction, accepts

FETCH_SHA256 = '0f06204497486a5bc742ea4b06638ec77fc5a01bd4e5c0aec0fa916b4fe371d3'
INVALID = '__invalid_native_output__'
DEPENDENCIES = {'progress':'1.5', 'tabulate':'0.9.0', 'textdistance':'4.1.5',
                'jiwer':'2.0.0', 'numpy':'1.26.4', 'python-Levenshtein':'0.12.2'}


def load_official_metrics(repository):
    repository = Path(repository).resolve()
    if file_hash(repository/'FETCH.json') != FETCH_SHA256:
        raise ValueError('official scorer source manifest pin changed')
    fetch = json.loads((repository/'FETCH.json').read_text())
    for name, digest in fetch['files'].items():
        if file_hash(repository/name) != digest: raise ValueError('official scorer source changed')
    versions = {k: importlib.metadata.version(k) for k in DEPENDENCIES}
    if versions != DEPENDENCIES: raise ValueError('official metric dependencies changed')
    source = repository/'scripts/evaluation'
    sys.path.insert(0, str(source))
    import metrics
    if Path(metrics.__file__).resolve() != source/'metrics/__init__.py':
        raise ValueError('another metrics package shadows official implementation')
    return metrics.ErrorMetric


def prepare_predictions(requests, targets, predictions, taxonomy):
    by_id = {r['id']: r for r in requests}; gold = {r['id']: r for r in targets}
    if (not by_id or len(by_id) != len(requests) or len(gold) != len(targets)
            or not set(by_id) <= set(gold)):
        raise ValueError('nonempty unique requests and covering targets required')
    if len({r['split'] for r in requests}) != 1: raise ValueError('single split required')
    taxonomy_sha = hashlib.sha256(canonical(taxonomy)).hexdigest()
    filenames = set()
    for uid, request in by_id.items():
        digest = hashlib.sha256(canonical(request)).hexdigest()
        if gold[uid]['request_sha256'] != digest or request['input']['schema_sha256'] != taxonomy_sha:
            raise ValueError('gold/request/taxonomy custody changed')
        filename = gold[uid]['metadata']['file']
        if filename in filenames: raise ValueError('duplicate recording filename')
        filenames.add(filename)
        target = parse_prediction(json.dumps(gold[uid]['target']))
        if not accepts(target, taxonomy): raise ValueError('invalid gold taxonomy')
    found = {}
    for row in predictions:
        uid = row['example_id']
        if uid not in by_id or uid in found or type(row['replicate']) is not int or row['replicate'] != 0:
            raise ValueError('unknown/duplicate prediction or nonzero replicate')
        if row['request_sha256'] != gold[uid]['request_sha256']:
            raise ValueError('prediction request binding changed')
        found[uid] = row
    if set(found) != set(by_id): raise ValueError('complete prediction coverage required')
    output = []; format_valid = schema_valid = 0
    for uid in by_id:
        try:
            value = parse_prediction(found[uid]['raw_text']); valid = True
        except (ValueError, TypeError):
            value = {'scenario': INVALID, 'action': INVALID, 'entities': []}; valid = False
        format_valid += valid
        schema_valid += valid and accepts(value, taxonomy)
        # Well-formed out-of-taxonomy outputs retain official false positives.
        output.append({'file': gold[uid]['metadata']['file'], **value})
    return output, {'examples': len(by_id), 'predictions': len(found),
                    'format_valid': format_valid, 'schema_valid': schema_valid,
                    'malformed': len(by_id)-format_valid, 'coverage': 1.}


def score_native(requests, targets, predictions, taxonomy, metric_factory):
    normalized, report = prepare_predictions(requests, targets, predictions, taxonomy)
    gold = {r['id']: r for r in targets}
    metrics = {k: metric_factory.get_instance('f1', average='micro')
               for k in ('scenario', 'action', 'joint_intent')}
    metrics['entity_exact'] = metric_factory.get_instance('span_f1', average='micro')
    for distance in ('word', 'char'):
        metrics[distance] = metric_factory.get_instance('span_distance_f1', average='micro', distance=distance)
    for request, prediction in zip(requests, normalized):
        target = gold[request['id']]['target']
        for key in ('scenario', 'action'): metrics[key](target[key], prediction[key])
        metrics['joint_intent'](target['scenario']+'_'+target['action'],
                                prediction['scenario']+'_'+prediction['action'])
        for key in ('entity_exact', 'word', 'char'): metrics[key](target['entities'], prediction['entities'])
    results = {k: metric.get_metric() for k, metric in metrics.items()}
    combined = metric_factory.get_instance('slu_f1', average='micro')
    combined(results['word']); combined(results['char']); results['slu_f1'] = combined.get_metric()
    report.update(metrics=results,
        malformed_policy='complete ID retained; invalid scenario/action sentinel and zero predicted entities; no output repair; official entity metrics give no accuracy credit for empty-empty entities',
        metric_columns=['precision', 'recall', 'f1', 'tp', 'fp', 'fn'],
        scope='official micro metrics on complete selected recording population; see split and full-official-split flag')
    return report, normalized
