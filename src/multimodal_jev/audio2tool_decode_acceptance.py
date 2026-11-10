"""Offline decode artifact checks; caller separately authenticates execution."""
import json
import math
from pathlib import Path

from .audio2tool_decode_profile import summarize
from .baseline import file_hash


def audit_records(run, preflight_run, decode):
    """Bind all natural-stop outputs to accepted real HF preflight inputs.

    Neither this function nor an arbitrary preflight path proves container or
    source provenance. The external coordinator must authenticate both runs.
    No labels, inference, waveform replay, or throughput extrapolation.
    """
    run, preflight_run = Path(run), Path(preflight_run)
    read = lambda p: json.loads(p.read_text())
    result = read(run/'result.json'); identity = read(run/'identity.json')
    preflight = read(preflight_run/'result.json'); before = read(preflight_run/'identity.json')
    if (preflight['status'] != 'preflight_passed' or preflight['model_loaded'] is not False
            or preflight['identity_sha256'] != file_hash(preflight_run/'identity.json')
            or before['preflight_only'] is not True or before['capacity'] is not None
            or read(preflight_run/'terminal.json')['exit_code'] != 0):
        raise ValueError('complete real input preflight required')
    if (result['status'] != 'profile_complete' or result['model_loaded'] is not True
            or result['identity_sha256'] != file_hash(run/'identity.json')
            or read(run/'terminal.json') != {'exit_code': 0}
            or identity['preflight_only'] is not False or not identity['capacity']
            or identity['sampling'] != dict(temperature=0., max_tokens=1024, n=1, seed=17)
            or any(d['targets_read'] is not False or d['weight_updates'] != 0
                   for d in (result, identity, preflight, before))):
        raise ValueError('complete frozen full-decode run required')
    for key in ('versions', 'engine', 'model', 'asset_audit_sha256', 'model_files', 'sources',
                'cases', 'sampling', 'input_pins', 'protocol'):
        if identity[key] != before[key]: raise ValueError('accepted preflight identity changed: '+key)
    names = identity['cases']; rows = result['records']; previous = preflight['records']
    if len(names) != 16 or len(set(names)) != 16 or len(rows) != 16 or len(previous) != 16:
        raise ValueError('all sixteen boundary cases required')
    fixed = {'identity.json', 'result.json', 'terminal.json', 'progress.json', 'effective-cache.json'}
    expected = {name+suffix for name in names for suffix in ('.json', '.attempt.json')}
    if {p.name for p in run.glob('*.json')} - fixed != expected:
        raise ValueError('exact complete boundary artifact roster required')
    hashes = {}
    for name, record, prior in zip(names, rows, previous, strict=True):
        path = run/(name+'.json'); attempt_path = run/(name+'.attempt.json')
        attempt = read(attempt_path)
        if (record != read(path) or prior != read(preflight_run/(name+'.json'))
                or record.get('error') or record['name'] != name or prior['name'] != name
                or prior['model_loaded'] is not False
                or set(record) != set(attempt)|{'model_loaded', 'engine_seconds', 'response'}
                or any(record.get(k) != v for k,v in attempt.items())):
            raise ValueError('attempt, preflight or raw record differs')
        fields = {'name','family','id','decision_id','prompt_sha256','audio_receipts',
                  'input_tokens','input_ids_sha256','preparation_seconds'}
        if set(attempt) != fields or any(attempt[k] != prior[k] for k in fields-{'preparation_seconds'}):
            raise ValueError('actual prompt/audio/token input differs from preflight')
        response = record['response']; tokens = response['token_ids']; reason = response['finish_reason']
        if (set(response) != {'raw_text','error','token_ids','finish_reason','length_limited'}
                or response['error'] is not None or type(response['raw_text']) is not str
                or type(tokens) is not list or not 1 <= len(tokens) <= 1024
                or any(type(t) is not int or t < 0 for t in tokens)
                or reason not in ('stop','length') or response['length_limited'] is not (reason == 'length')
                or (reason == 'length' and len(tokens) != 1024)
                or decode(tokens) != response['raw_text']):
            raise ValueError('raw decoded output or natural-stop contract differs')
        hashes[name] = dict(attempt_sha256=file_hash(attempt_path), record_sha256=file_hash(path))
    summary = summarize(rows, names)
    if summary != result['summary']: raise ValueError('reported boundary timing differs from raw records')
    for name in ('elapsed_seconds', 'engine_load_seconds'):
        value = result[name]
        if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
            raise ValueError('positive finite runtime duration required')
    measured = result['engine_load_seconds'] + sum(r['preparation_seconds']+r['engine_seconds'] for r in rows)
    if measured > result['elapsed_seconds'] + 1e-6: raise ValueError('inconsistent total runtime')
    cache = read(run/'effective-cache.json')
    if (type(cache['configuration_exposed']) is not bool
            or set(cache['fields']) != {'mm_processor_cache_gb','mm_processor_cache_type','mm_shm_cache_max_object_size_mb'}):
        raise ValueError('actual cache configuration observation required')
    return dict(records=16, result_sha256=file_hash(run/'result.json'),
                identity_sha256=file_hash(run/'identity.json'), artifacts=hashes,
                cache_observation=cache, recomputed_summary=summary,
                scope='Saved outputs and preflight input identity only; caller must independently bind actual containers, source, model, capacities and tokenizer. No population throughput estimate or task score.')
