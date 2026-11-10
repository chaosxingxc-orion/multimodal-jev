"""Bounded-file, append-only ledger for independent full-bank Jev generation.

This binds artifacts, not model execution: actual audio/token/container checks
remain the responsibility of the runtime and an independent actor acceptance.
No resume, repair, automatic replay, or fabricated unattempted records.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from .audio2tool_inference import digest, write_once
from .audio2tool_score import typed
from .baseline import file_hash

PROTOCOL = 'audio2tool-jev-sharded-ledger-v1'


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _header(requests, bank, shard_decisions):
    if type(shard_decisions) is not int or not 1 <= shard_decisions <= 65536:
        raise ValueError('bounded positive shard size required')
    bank_sha = digest(bank)
    ids = [d['id'] for d in bank['decisions']]
    if not ids or any(type(key) is not str or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError('unique nonempty decision roster required')
    roster = []
    for r in requests:
        if (type(r['id']) is not str or not r['id'] or r['input']['task_bank_sha256'] != bank_sha):
            raise ValueError('bank-bound input roster required')
        roster.append(dict(id=r['id'], request_sha256=digest(r)))
    if not roster or len({r['id'] for r in roster}) != len(roster):
        raise ValueError('unique nonempty release roster required')
    return dict(protocol=PROTOCOL, roster=roster, decision_ids=ids, bank_sha256=bank_sha,
                shard_decisions=shard_decisions, max_output_tokens=1024,
                total_decisions=len(roster) * len(ids))


def _validate_response(response, attempt, seconds):
    if (set(response) != {'raw_text', 'error', 'token_ids', 'finish_reason', 'length_limited',
                         'input_tokens', 'input_ids_sha256'}
            or type(response['raw_text']) is not str
            or type(response['token_ids']) is not list
            or type(response['length_limited']) is not bool
            or type(response['input_tokens']) is not int
            or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
            or response['input_tokens'] != attempt['input_tokens']
            or response['input_ids_sha256'] != attempt['input_ids_sha256']):
        raise ValueError('original input identity and finite terminal timing required')
    if response['error'] is not None:
        if (type(response['error']) is not str or not response['error'] or response['raw_text'] != ''
                or response['token_ids'] != [] or response['finish_reason'] != 'runtime_error'
                or response['length_limited']):
            raise ValueError('explicit actual runtime failure required')
    elif (any(type(i) is not int or i < 0 for i in response['token_ids'])
            or not 1 <= len(response['token_ids']) <= 1024
            or response['finish_reason'] not in ('stop', 'length')
            or response['length_limited'] != (response['finish_reason'] == 'length')
            or (response['length_limited'] and len(response['token_ids']) != 1024)):
        raise ValueError('unchanged raw 1024-token generation contract required')


class DecisionLedger:
    def __init__(self, root, requests, bank, *, shard_decisions=4096):
        self.header = _header(requests, bank, shard_decisions)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        write_once(self.root / 'manifest.json', self.header)
        _sync_directory(self.root); _sync_directory(self.root.parent)
        self.completed = 0; self.events = 0; self.context = None; self.pending = None
        self.chain = digest(self.header); self.handle = None; self.shards = []
        self.failed = False; self.closed = False; self.length_limited = 0

    def _ready(self):
        if self.closed or self.failed:
            raise ValueError('closed or failed ledger cannot continue')

    def _close_shard(self):
        if self.handle is not None:
            self.handle.close(); self.handle = None

    def _append(self, event):
        self._ready()
        core = dict(index=self.events, previous=self.chain, event=event)
        envelope = dict(core, sha256=digest(core))
        try:
            if self.handle is None:
                name = f'part-{len(self.shards):06d}.jsonl'
                self.handle = (self.root / name).open('x', encoding='utf-8')
                self.shards.append(name); _sync_directory(self.root)
            self.handle.write(json.dumps(envelope, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            self.handle.flush(); os.fsync(self.handle.fileno())
        except BaseException:
            # A write/fsync may have reached storage even if the caller saw an
            # exception. Poison this writer instead of retrying an ambiguous IO.
            self.failed = True; self._close_shard()
            raise
        self.chain = envelope['sha256']; self.events += 1
        return envelope['sha256']

    def begin(self, request, decision_id, prompt, receipts, expected_ids):
        self._ready()
        if self.pending is not None or self.completed >= self.header['total_decisions']:
            raise ValueError('one outstanding attempt within complete roster required')
        width = len(self.header['decision_ids']); row, column = divmod(self.completed, width)
        expected = self.header['roster'][row]
        if (request['id'] != expected['id'] or digest(request) != expected['request_sha256']
                or decision_id != self.header['decision_ids'][column]
                or type(prompt) is not str or type(receipts) is not list or not receipts
                or type(expected_ids) is not list or not expected_ids
                or any(type(i) is not int or i < 0 for i in expected_ids)):
            raise ValueError('exact ordered request/decision and actual prompt/audio/token evidence required')
        typed(receipts)
        context = dict(kind='context', row=row, **expected, input_sha256=digest(request['input']),
                       audio_receipts=receipts)
        if column and digest(context) != self.context:
            raise ValueError('audio receipts changed within a request')
        if self.completed % self.header['shard_decisions'] == 0:
            self._close_shard()
        if not column:
            self._append(context); self.context = digest(context)
        attempt = dict(kind='attempt', ordinal=self.completed, example_id=request['id'],
                       decision_id=decision_id, request_sha256=expected['request_sha256'],
                       context_sha256=self.context,
                       prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                       input_tokens=len(expected_ids), input_ids_sha256=digest(expected_ids))
        # Keep only the immutable digest of the return value: caller mutations
        # cannot silently alter the identity of a pending attempt.
        self._append(attempt); self.pending = digest(attempt)
        return attempt

    def finish(self, attempt, response, seconds):
        self._ready()
        if self.pending is None or digest(attempt) != self.pending:
            raise ValueError('exact outstanding attempt required')
        _validate_response(response, attempt, seconds)
        event = dict(kind='terminal', ordinal=self.completed, attempt_sha256=self.pending,
                     response=response, seconds=seconds)
        self._append(event)
        self.completed += 1; self.pending = None
        self.length_limited += response['length_limited']
        if response['error'] is not None:
            self.failed = True; self._close_shard()
        return event

    def failure(self, attempt, exc, seconds):
        return self.finish(attempt, dict(raw_text='', error=repr(exc), token_ids=[],
            finish_reason='runtime_error', length_limited=False,
            input_tokens=attempt['input_tokens'], input_ids_sha256=attempt['input_ids_sha256']), seconds)

    def complete(self):
        self._ready()
        if self.pending is not None or self.completed != self.header['total_decisions']:
            raise ValueError('all declared attempts and successful terminal records required')
        self._close_shard(); self.closed = True
        result = dict(protocol=PROTOCOL, manifest_sha256=file_hash(self.root / 'manifest.json'),
            events=self.events, decisions=self.completed, final_chain=self.chain,
            length_limited=self.length_limited,
            shards={name:file_hash(self.root / name) for name in self.shards})
        write_once(self.root / 'complete.json', result); _sync_directory(self.root)
        return result

    def close(self):
        self._close_shard(); self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate artifact JSON key')
            value[key] = item
        return value
    value = json.loads(text, object_pairs_hook=pairs)
    typed(value)
    return value


def records(root, requests, bank):
    """Read a complete ledger against the caller's independently fixed roster.

    Must exhaust this iterator before accepting any score/export. Hashes detect
    alteration relative to completion; they are not a signature or proof that
    recorded audio and tokens came from an actual engine.
    """
    root = Path(root)
    header = _strict_json((root / 'manifest.json').read_text())
    if typed(header) != typed(_header(requests, bank, header['shard_decisions'])):
        raise ValueError('independently declared roster and task bank required')
    complete = _strict_json((root / 'complete.json').read_text())
    complete_sha = file_hash(root / 'complete.json')
    if (set(complete) != {'protocol', 'manifest_sha256', 'events', 'decisions', 'final_chain', 'length_limited', 'shards'}
            or complete['protocol'] != PROTOCOL
            or complete['manifest_sha256'] != file_hash(root / 'manifest.json')):
        raise ValueError('unchanged complete ledger identity required')
    total = header['total_decisions']; width = len(header['decision_ids'])
    count = (total + header['shard_decisions'] - 1) // header['shard_decisions']
    names = [f'part-{i:06d}.jsonl' for i in range(count)]
    if set(complete['shards']) != set(names) or {p.name for p in root.iterdir()} != set(names) | {'manifest.json', 'complete.json'}:
        raise ValueError('exact complete shard population required')
    ordinal = 0; event_index = 0; chain = digest(header); context = None; pending = None; limited = 0
    for shard_index, name in enumerate(names):
        path = root / name
        if file_hash(path) != complete['shards'][name]:
            raise ValueError('shard content changed')
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.endswith('\n'):
                    raise ValueError('incomplete journal line')
                envelope = _strict_json(line)
                if set(envelope) != {'index', 'previous', 'event', 'sha256'}:
                    raise ValueError('exact hash-chain envelope required')
                core = {k:envelope[k] for k in ('index', 'previous', 'event')}
                if (type(core['index']) is not int or core['index'] != event_index or core['previous'] != chain
                        or digest(core) != envelope['sha256'] or ordinal >= total
                        or ordinal // header['shard_decisions'] != shard_index):
                    raise ValueError('ordered complete journal chain required')
                event = core['event']; row, column = divmod(ordinal, width); r = requests[row]
                if pending is not None:
                    if (set(event) != {'kind', 'ordinal', 'attempt_sha256', 'response', 'seconds'}
                            or event['kind'] != 'terminal' or type(event['ordinal']) is not int
                            or event['ordinal'] != ordinal or event['attempt_sha256'] != digest(pending)):
                        raise ValueError('each attempt requires its exact terminal event')
                    _validate_response(event['response'], pending, event['seconds'])
                    if event['response']['error'] is not None:
                        raise ValueError('runtime failure cannot complete full inference')
                    limited += event['response']['length_limited']
                    yield dict(attempt=pending, response=event['response'], seconds=event['seconds'],
                               context=context)
                    pending = None; ordinal += 1
                elif context is None or context['row'] != row:
                    if (set(event) != {'kind', 'row', 'id', 'request_sha256', 'input_sha256', 'audio_receipts'}
                            or event['kind'] != 'context' or type(event['row']) is not int or event['row'] != row
                            or column != 0 or event['id'] != r['id'] or event['request_sha256'] != digest(r)
                            or event['input_sha256'] != digest(r['input'])
                            or type(event['audio_receipts']) is not list or not event['audio_receipts']):
                        raise ValueError('one original input context per release record required')
                    context = event
                else:
                    if (set(event) != {'kind', 'ordinal', 'example_id', 'decision_id', 'request_sha256',
                                      'context_sha256', 'prompt_sha256', 'input_tokens', 'input_ids_sha256'}
                            or event['kind'] != 'attempt' or type(event['ordinal']) is not int or event['ordinal'] != ordinal
                            or event['example_id'] != r['id'] or event['request_sha256'] != digest(r)
                            or event['decision_id'] != header['decision_ids'][column]
                            or event['context_sha256'] != digest(context)
                            or type(event['input_tokens']) is not int or event['input_tokens'] <= 0
                            or any(type(event[k]) is not str or len(event[k]) != 64 or
                                   any(c not in '0123456789abcdef' for c in event[k])
                                   for k in ('input_ids_sha256', 'prompt_sha256'))):
                        raise ValueError('ordered input-bound decision attempt required')
                    pending = event
                chain = envelope['sha256']; event_index += 1
        if file_hash(path) != complete['shards'][name]:
            raise ValueError('shard changed during verification')
    if (pending is not None or ordinal != total or complete['final_chain'] != chain
            or typed([complete['events'], complete['decisions'], complete['length_limited']]) != typed([event_index, ordinal, limited])
            or complete['manifest_sha256'] != file_hash(root / 'manifest.json')
            or complete_sha != file_hash(root / 'complete.json')):
        raise ValueError('complete final counts and chain required')


def export_predictions(root, requests, bank, output):
    """Export raw scorer rows outside the journal, accepting only full exhaustion."""
    output = Path(output)
    if output.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('projection must stay outside immutable ledger')
    count = 0
    with output.open('x', encoding='utf-8') as handle:
        for r in records(root, requests, bank):
            attempt, response = r['attempt'], r['response']
            p = {k:attempt[k] for k in ('example_id', 'decision_id', 'request_sha256')}
            p.update(raw_text=response['raw_text'], error=response['error'])
            handle.write(json.dumps(p, ensure_ascii=False, allow_nan=False) + '\n'); count += 1
        handle.flush(); os.fsync(handle.fileno())
    _sync_directory(output.parent)
    return dict(decisions=count, predictions_sha256=file_hash(output), actor_execution_verified=False)
