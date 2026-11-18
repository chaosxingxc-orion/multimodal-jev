"""Bulk IPC with bounded, deduplicated model batches under one service owner.

Deduplication is local to one command/policy. It changes neither query exposure
nor downstream G16 sampling: one record is returned for every original row.
"""
import copy
from types import SimpleNamespace

from multimodal_jev.context_rollout_batch import bind_rollout
from multimodal_jev.span_rollout_rpc import send, receive
from tools.reports.context_bulk_rpc_candidate import read_rows, validate_rows
from tools.reports.context_batched_request_candidate import prepare_request


def plan_batches(engine, rows, audio_token_id):
    validate_rows(rows)
    if (type(engine.max_batch_queries) is not int or not 1 <= engine.max_batch_queries <= 128
            or type(engine.max_batch_tokens) is not int or engine.max_batch_tokens < 1):
        raise ValueError('explicit positive model batch capacities required')
    # Validate the entire command before any model execution, including final rows.
    prepared = [prepare_request(engine, row, audio_token_id, SimpleNamespace) for row in rows]
    unique = {}; restore = []
    for index, p in enumerate(prepared):
        rid = p['request_id']; restore.append(rid)
        if len(p['ids']) > engine.max_batch_tokens:
            raise ValueError('individual complete request exceeds model batch token capacity')
        if rid in unique:
            if p['request_identity'] != prepared[unique[rid]]['request_identity']:
                raise ValueError('request identity collision')
        else:
            unique[rid] = index
    # Stable length order; no labels or rewards enter packing.
    indices = sorted(unique.values(), key=lambda i: (len(prepared[i]['ids']), i))
    batches = []; current = []; tokens = 0
    for index in indices:
        length = len(prepared[index]['ids'])
        if current and (len(current) == engine.max_batch_queries or tokens + length > engine.max_batch_tokens):
            batches.append(current); current = []; tokens = 0
        current.append(index); tokens += length
    if current:
        batches.append(current)
    return prepared, restore, batches


def score_rows(engine, rows, *, policy_id, audio_token_id):
    if engine.bundle['policy_id'] != policy_id:
        raise ValueError('stale service policy')
    prepared, restore, batches = plan_batches(engine, rows, audio_token_id)
    records = {}
    for indices in batches:
        if engine.bundle['policy_id'] != policy_id:
            raise ValueError('policy changed within command')
        values = engine.score_many([rows[i] for i in indices], audio_token_id=audio_token_id)
        if type(values) is not list or len(values) != len(indices):
            raise ValueError('incomplete model batch')
        for index, value in zip(indices, values, strict=True):
            bind_rollout(value, **rows[index], policy_id=policy_id, group_size=2, seed=0)
            if value['request_id'] != prepared[index]['request_id']:
                raise ValueError('wrong model request identity')
            records[value['request_id']] = value
    if engine.bundle['policy_id'] != policy_id:
        raise ValueError('policy changed before reply')
    result = [copy.deepcopy(records[rid]) for rid in restore]
    receipt = dict(rows=len(rows), unique_rows=len(records), encode_calls=len(batches),
                   batch_queries=[len(b) for b in batches],
                   batch_tokens=[sum(len(prepared[i]['ids']) for i in b) for b in batches],
                   policy_id=policy_id, scope='encode calls; not scheduler model-forward count')
    return result, receipt


def serve_connection(connection, engine, exchange):
    sequence = 0
    while True:
        try:
            command = receive(connection)
        except EOFError:
            return
        try:
            if type(command.get('sequence')) is not int or command['sequence'] != sequence:
                raise ValueError('exact command sequence required')
            operation = command.get('operation')
            if operation == 'score_many':
                if set(command) != {'sequence','operation','policy_id','path','sha256','audio_token_id'}:
                    raise ValueError('label-free bulk command required')
                rows = read_rows(command['path'], command['sha256'], exchange)
                result, receipt = score_rows(engine, rows, policy_id=command['policy_id'],
                                             audio_token_id=command['audio_token_id'])
                engine.last_batch_receipt = receipt
            elif operation == 'switch':
                if set(command) != {'sequence','operation','previous_policy_id','snapshot','policy_id','adapter'}:
                    raise ValueError('complete switch command required')
                if engine.bundle['policy_id'] != command['previous_policy_id']:
                    raise ValueError('stale switch source')
                result = engine.switch(**{k:command[k] for k in ('snapshot','policy_id','adapter')})
            elif operation == 'close':
                if set(command) != {'sequence','operation'}:
                    raise ValueError('invalid close')
                send(connection, dict(sequence=sequence, ok=True, result=None)); return
            else:
                raise ValueError('unknown batched service operation')
            send(connection, dict(sequence=sequence, ok=True, result=result)); sequence += 1
        except Exception as exc:
            send(connection, dict(sequence=sequence, ok=False, error=repr(exc)))
            raise
