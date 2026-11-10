"""Independent raw-output audit; does not reuse rollout capture/conversion.

No labels or task-quality thresholds. The external caller supplies verified
processor evidence, tokenizer, policy and effective sampling configuration.
"""
import json
import math
from pathlib import Path

from .baseline import file_hash
from .sfc_reward_batch import digest, valid_hash


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def audit_records(run, rows, checks, tokenizer, *, policy_sha256, instruction_sha256, sampling):
    """Require the entire original 78 x 2 roster and three artifacts per request."""
    run = Path(run)
    if not valid_hash(policy_sha256) or not valid_hash(instruction_sha256):
        raise ValueError('verified policy/instruction pins required')
    roster = [(r, d['id']) for r in rows for d in r['input']['decisions']]
    if (len(rows) != 6 or len({r['id'] for r in rows}) != 6 or len(roster) != 78
            or len({r['source_group'] for r in rows}) != 4
            or any(r['split'] != 'train_id' for r in rows)
            or set(checks) != {(r['id'], d) for r, d in roster}):
        raise ValueError('exact complete TRAIN78 engineering roster required')
    if sampling['n'] != 2 or sampling['max_tokens'] != 1024 or sampling['logprobs'] != 0:
        raise ValueError('registered K=2 probability capture required')
    directory = run / 'requests'
    expected_paths = {digest((r['id'], d)) + suffix for r, d in roster
                      for suffix in ('.intent.json', '.raw.json', '.result.json')}
    if {p.name for p in directory.iterdir()} != expected_paths:
        raise ValueError('exact intent/raw/result file roster required')
    all_records = []
    costs = []
    raw_hashes = {}
    for row, decision in roster:
        key = row['id'], decision
        name = digest(key)
        paths = {kind: directory / (name + '.' + kind + '.json') for kind in ('intent','raw','result')}
        intent, raw, result = (read_json(paths[k]) for k in ('intent','raw','result'))
        if (intent['example_id'] != key[0] or intent['decision_id'] != key[1]
                or intent['policy_sha256'] != policy_sha256 or intent['sampling'] != sampling
                or type(intent['started_unix']) not in (int,float)
                or not math.isfinite(intent['started_unix'])):
            raise ValueError('dispatch intent binding changed')
        seconds = result['seconds']
        if type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('finite nonnegative request cost required')
        costs.append(seconds)
        ids = result['input_ids']
        if (type(ids) is not list or not ids
                or any(type(t) is not int or not 0 <= t < len(tokenizer) for t in ids)
                or digest(ids) != checks[key]['input_ids_sha256'] or len(ids) + 1024 > 4096):
            raise ValueError('expanded input token evidence changed')
        if len(raw['outputs']) != 1:
            raise ValueError('one raw request output required')
        output = raw['outputs'][0]
        if output['finished'] is not True or output['prompt_token_ids'] != ids or len(output['outputs']) != 2:
            raise ValueError('complete raw engine output with identical prompt required')
        rebuilt = {}
        for answer in output['outputs']:
            index = answer['index']; tokens = answer['token_ids']; text = answer['text']
            if (type(index) is not int or index not in (0,1) or index in rebuilt
                    or type(tokens) is not list or not 1 <= len(tokens) <= 1024
                    or any(type(t) is not int or not 0 <= t < len(tokenizer) for t in tokens)
                    or type(text) is not str or answer['finish_reason'] not in ('stop','length')):
                raise ValueError('invalid raw replicate or termination')
            if tokenizer.decode(tokens, skip_special_tokens=True) != text:
                raise ValueError('raw text differs from actual sampled tokens')
            entries = answer['logprobs']
            if type(entries) is not list or len(entries) != len(tokens):
                raise ValueError('raw token/logprob alignment differs')
            probs = []
            for token, entry in zip(tokens, entries, strict=True):
                if type(entry) is not dict or str(token) not in entry:
                    raise ValueError('raw sampled-token probability missing')
                value = entry[str(token)]
                if type(value) not in (float,int) or not math.isfinite(value) or value > 1e-5:
                    raise ValueError('invalid raw sampled-token logprob')
                probs.append(float(value))
            rebuilt[index] = dict(example_id=row['id'],decision_id=decision,request_sha256=digest(row),
                replicate=index,policy_sha256=policy_sha256,instruction_sha256=instruction_sha256,
                prompt_sha256=checks[key]['prompt_sha256'],input_ids_sha256=checks[key]['input_ids_sha256'],
                raw_text=text,token_ids=tokens,log_probs=probs,finish_reason=answer['finish_reason'])
        expected = [rebuilt[i] for i in (0,1)]
        if result['records'] != expected:
            raise ValueError('converted records differ from independent raw reconstruction')
        all_records.extend(expected)
        raw_hashes.update({str(p.relative_to(run)): file_hash(p) for p in paths.values()})
    records = [json.loads(line) for line in (run/'records.jsonl').open()]
    if records != all_records:
        raise ValueError('merged records differ from full ordered raw population')
    summary = read_json(run/'result.json')
    if (summary['status'] != 'passed' or summary['model_loaded'] is not True
            or summary['weight_updates'] != 0
            or (summary['bundles'],summary['decisions'],summary['responses']) != (6,78,156)
            or summary['records_sha256'] != file_hash(run/'records.jsonl')
            or summary['identity_sha256'] != file_hash(run/'identity.json')
            or read_json(run/'terminal.json')['exit_code'] != 0):
        raise ValueError('successful full sampling result binding required')
    return dict(bundles=6,decisions=78,responses=156,
        response_tokens=sum(len(r['token_ids']) for r in records),
        length_limited=sum(r['finish_reason']=='length' for r in records),
        request_seconds=math.fsum(costs),records_sha256=file_hash(run/'records.jsonl'),
        request_artifacts_sha256=digest(raw_hashes),weight_updates=0,
        scope='Independent raw probability/identity audit; no quality scoring, HF forward parity or training')
