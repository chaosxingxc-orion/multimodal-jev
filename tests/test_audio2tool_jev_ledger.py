import copy
import json
from pathlib import Path

import pytest

from multimodal_jev.audio2tool_inference import digest
from multimodal_jev.audio2tool_jev_ledger import DecisionLedger, export_predictions, records
from multimodal_jev.baseline import file_hash


def fixture():
    bank = dict(decisions=[dict(id=k) for k in ('p', 'a', 'c')])
    rows = [dict(id=str(i), input=dict(task_bank_sha256=digest(bank), audio='fixture-' + str(i))) for i in range(3)]
    receipts = [dict(path='fixture.wav', actor_waveform_sha256='0' * 64, actor_samples=160)]
    return bank, rows, receipts


def response(attempt, raw='not repaired: 中文'):
    return dict(raw_text=raw, error=None, token_ids=[7, 8], finish_reason='stop', length_limited=False,
                input_tokens=attempt['input_tokens'], input_ids_sha256=attempt['input_ids_sha256'])


def filled(tmp_path, shard=2):
    bank, rows, receipts = fixture(); root = tmp_path / 'ledger'
    with DecisionLedger(root, rows, bank, shard_decisions=shard) as ledger:
        for row in rows:
            for decision in bank['decisions']:
                a = ledger.begin(row, decision['id'], 'prompt', receipts, [1, 2, 3])
                ledger.finish(a, response(a), .25)
        completed = ledger.complete()
    return root, bank, rows, completed


def test_rotation_within_request_and_full_raw_projection(tmp_path):
    root, bank, rows, completed = filled(tmp_path)
    observed = list(records(root, rows, bank))
    assert len(observed) == 9 and len(completed['shards']) == 5
    assert completed['events'] == 3 + 2 * 9
    assert [r['attempt']['decision_id'] for r in observed] == ['p', 'a', 'c'] * 3
    assert all(r['response']['raw_text'] == 'not repaired: 中文' for r in observed)
    out = tmp_path / 'predictions.jsonl'
    report = export_predictions(root, rows, bank, out)
    assert report['decisions'] == 9 and report['actor_execution_verified'] is False
    assert [json.loads(l)['raw_text'] for l in out.read_text().splitlines()] == ['not repaired: 中文'] * 9
    with pytest.raises(FileExistsError): export_predictions(root, rows, bank, out)
    with pytest.raises(ValueError, match='outside'): export_predictions(root, rows, bank, root / 'projection.jsonl')


@pytest.mark.parametrize('stage', ['empty', 'attempt', 'terminal', 'runtime_failure'])
def test_interruptions_and_runtime_failures_never_complete_or_resume(tmp_path, stage):
    bank, rows, receipts = fixture(); root = tmp_path / 'ledger'
    with DecisionLedger(root, rows, bank) as ledger:
        if stage != 'empty':
            a = ledger.begin(rows[0], 'p', 'prompt', receipts, [1])
            if stage == 'terminal': ledger.finish(a, response(a), 0)
            if stage == 'runtime_failure':
                ledger.failure(a, RuntimeError('engine crashed'), 1)
                with pytest.raises(ValueError): ledger.begin(rows[0], 'a', 'prompt', receipts, [1])
        with pytest.raises(ValueError): ledger.complete()
    assert not (root / 'complete.json').exists()
    with pytest.raises(FileExistsError): DecisionLedger(root, rows, bank)
    with pytest.raises(FileNotFoundError): list(records(root, rows, bank))


def test_duplicate_calls_and_mutated_attempts_rejected(tmp_path):
    bank, rows, receipts = fixture()
    with DecisionLedger(tmp_path / 'ledger', rows, bank) as ledger:
        with pytest.raises(ValueError): ledger.begin(rows[0], 'a', 'prompt', receipts, [1])
        a = ledger.begin(rows[0], 'p', 'prompt', receipts, [1])
        with pytest.raises(ValueError): ledger.begin(rows[0], 'p', 'prompt', receipts, [1])
        modified = dict(a, prompt_sha256='a' * 64)
        with pytest.raises(ValueError): ledger.finish(modified, response(modified), .1)
        ledger.finish(a, response(a), .1)
        with pytest.raises(ValueError): ledger.finish(a, response(a), .1)
        wrong = copy.deepcopy(receipts); wrong[0]['actor_samples'] = 320
        with pytest.raises(ValueError): ledger.begin(rows[0], 'a', 'prompt', wrong, [1])
        mutated = copy.deepcopy(rows[0]); mutated['input']['audio'] = 'other'
        with pytest.raises(ValueError): ledger.begin(mutated, 'a', 'prompt', receipts, [1])


@pytest.mark.parametrize('fault', ['wrong_ids', 'short_length', 'boolean_token', 'empty_tokens', 'bad_time', 'false_error', 'extra'])
def test_bad_terminal_contract_rejected(tmp_path, fault):
    bank, rows, receipts = fixture()
    with DecisionLedger(tmp_path / 'ledger', rows, bank) as ledger:
        a = ledger.begin(rows[0], 'p', 'prompt', receipts, [1]); r = response(a); seconds = .1
        if fault == 'wrong_ids': r['input_ids_sha256'] = 'changed'
        if fault == 'short_length': r.update(finish_reason='length', length_limited=True)
        if fault == 'boolean_token': r['token_ids'] = [True]
        if fault == 'empty_tokens': r['token_ids'] = []
        if fault == 'bad_time': seconds = float('nan')
        if fault == 'false_error': r['error'] = False
        if fault == 'extra': r['fixed_json'] = '{}'
        with pytest.raises(ValueError): ledger.finish(a, r, seconds)
        assert ledger.completed == 0 and ledger.pending is not None


def rewrite_chain(root, mutate):
    header = json.loads((root / 'manifest.json').read_text())
    complete = json.loads((root / 'complete.json').read_text()); chain = digest(header)
    for name in complete['shards']:
        path = root / name; envelopes = [json.loads(l) for l in path.read_text().splitlines()]
        for e in envelopes:
            mutate(e['event'])
            e['previous'] = chain
            e['sha256'] = digest({k:e[k] for k in ('index', 'previous', 'event')}); chain = e['sha256']
        path.write_text(''.join(json.dumps(e) + '\n' for e in envelopes))
        complete['shards'][name] = file_hash(path)
    complete['final_chain'] = chain
    (root / 'complete.json').write_text(json.dumps(complete))


@pytest.mark.parametrize('fault', ['truncated', 'modified', 'extra_shard', 'missing_shard', 'wrong_roster',
                                  'wrong_bank', 'wrong_count', 'rehashed_wrong_order', 'rehashed_wrong_context'])
def test_full_reader_rejects_missing_or_altered_evidence(tmp_path, fault):
    root, bank, rows, complete = filled(tmp_path)
    shard = root / 'part-000001.jsonl'
    if fault == 'truncated': shard.write_bytes(shard.read_bytes()[:-1])
    if fault == 'modified': shard.write_text(shard.read_text().replace('中文', 'fixed'))
    if fault == 'extra_shard': (root / 'part-000005.jsonl').write_text('')
    if fault == 'missing_shard': shard.unlink()
    if fault == 'wrong_roster': rows[0]['input']['audio'] = 'other'
    if fault == 'wrong_bank': bank['decisions'].reverse()
    if fault == 'wrong_count':
        complete['decisions'] = 8; (root / 'complete.json').write_text(json.dumps(complete))
    if fault == 'rehashed_wrong_order':
        rewrite_chain(root, lambda e:e.update(decision_id='c') if e['kind'] == 'attempt' and e['ordinal'] == 0 else None)
    if fault == 'rehashed_wrong_context':
        rewrite_chain(root, lambda e:e.update(context_sha256='a' * 64) if e['kind'] == 'attempt' else None)
    with pytest.raises(ValueError): list(records(root, rows, bank))


def test_failed_fsync_poisons_writer_without_retry(tmp_path, monkeypatch):
    bank, rows, receipts = fixture()
    ledger = DecisionLedger(tmp_path / 'ledger', rows, bank)
    def fail(_): raise OSError('synthetic fsync error')
    monkeypatch.setattr('multimodal_jev.audio2tool_jev_ledger.os.fsync', fail)
    with pytest.raises(OSError): ledger.begin(rows[0], 'p', 'prompt', receipts, [1])
    assert ledger.handle is None
    with pytest.raises(ValueError): ledger.begin(rows[0], 'p', 'prompt', receipts, [1])
    with pytest.raises(ValueError): ledger.complete()


def test_ambiguous_terminal_fsync_cannot_be_retried(tmp_path, monkeypatch):
    bank, rows, receipts = fixture()
    with DecisionLedger(tmp_path / 'ledger', rows, bank) as ledger:
        a = ledger.begin(rows[0], 'p', 'prompt', receipts, [1])
        def fail(_): raise OSError('synthetic terminal fsync error')
        monkeypatch.setattr('multimodal_jev.audio2tool_jev_ledger.os.fsync', fail)
        with pytest.raises(OSError): ledger.finish(a, response(a), .1)
        assert ledger.handle is None
        with pytest.raises(ValueError): ledger.finish(a, response(a), .1)
        assert not (ledger.root / 'complete.json').exists()


def test_open_shard_descriptor_count_is_constant(tmp_path):
    bank, rows, receipts = fixture(); root = tmp_path / 'ledger'
    with DecisionLedger(root, rows, bank, shard_decisions=1) as ledger:
        for row in rows:
            for d in bank['decisions']:
                a = ledger.begin(row, d['id'], 'prompt', receipts, [1])
                links = []
                for fd in Path('/proc/self/fd').iterdir():
                    try: links.append(str(fd.resolve(strict=True)))
                    except FileNotFoundError: pass
                assert sum(link.startswith(str(root / 'part-')) for link in links) == 1
                ledger.finish(a, response(a), .1)
        ledger.complete()
        assert ledger.handle is None
