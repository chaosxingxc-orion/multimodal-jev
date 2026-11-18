"""Continuous scheduled RL driver over an already loaded distributed session.

All ranks call run together. Runtime callbacks own actual rollout, collective
batch delivery, BF16 export, independent probability checks and durable shards.
There is no model reload or optimizer construction inside this loop.
"""
import hashlib
import json
import time


def run(session, schedule, *, stop_update, collect, export, verify, checkpoint,
        checkpoint_interval, record, evaluate=None):
    """Execute the remaining declared prefix with exact schedule/session cursors.

    collect(slots,cursor,policy_id) returns this rank's authenticated DataProto;
    export(session,batch,metrics) publishes actual immutable adapter bytes;
    verify(policy_id,updates,batch) checks the new resident actor/export/rollout;
    checkpoint(session,cursor) durably binds master/Adam, policy and next cursor.
    Callback failures are collective and poison this session, including errors
    after an optimizer step. Resume must restore an independently accepted
    checkpoint into a new session; this function never rolls back or retries.
    """
    error = None
    if (type(stop_update) is not int or not session.completed_updates < stop_update <= schedule.total_updates
            or type(checkpoint_interval) is not int or checkpoint_interval < 1):
        error = 'bounded nonempty schedule and positive checkpoint interval required'
    session._agree('simple_loop_start:'+str(stop_update)+':'+str(checkpoint_interval)+':'+schedule.sha256,
                   'accepted', error)

    def collective(name, callback):
        error = None; result = None
        try:
            result = callback()
        except Exception as exc:
            error = repr(exc)
        except BaseException:
            session.phase = 'failed'
            raise
        try:
            session._agree('simple_loop_'+name, session.phase, error)
        except BaseException:
            session.phase = 'failed'
            raise
        return result

    while session.completed_updates < stop_update:
        step = session.completed_updates
        slots, cursor = collective('schedule', lambda: schedule.at(step))
        def validate_cursor():
            if (cursor['update'] != step or cursor['next_optimizer_update'] != step+1
                    or cursor['total_updates'] != schedule.total_updates
                    or cursor['schedule_sha256'] != schedule.sha256):
                raise ValueError('schedule cursor differs')
        collective('cursor', validate_cursor)
        cursor_digest = collective('cursor_digest', lambda: hashlib.sha256(json.dumps(
            dict(slots=slots, cursor=cursor), sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode()).hexdigest())
        collective('cursor_agreement:'+cursor_digest, lambda: None)
        behavior_policy = session.policy_id
        begin = time.monotonic()
        batch = collective('collect', lambda: collect(slots, cursor, behavior_policy))
        collect_seconds = time.monotonic()-begin
        begin = time.monotonic()
        metrics = session.update(batch)
        update_seconds = time.monotonic()-begin
        published = collective('export', lambda: export(session, batch, metrics))
        session.publish(published)
        begin = time.monotonic()
        receipt = session.accept(lambda policy, updates: verify(policy, updates, batch))
        verify_seconds = time.monotonic()-begin
        save = (session.completed_updates % checkpoint_interval == 0
                or session.completed_updates == stop_update
                or cursor['batch_index'] == 316)
        checkpoint_receipt = None
        if save:
            checkpoint_receipt = collective('checkpoint', lambda: checkpoint(session, cursor))
        report = dict(completed_updates=session.completed_updates,
            next_update=session.completed_updates, behavior_policy_id=behavior_policy,
            policy_id=session.policy_id, schedule_sha256=schedule.sha256,
            coefficient=cursor['coefficient'], real_requests=cursor['real_requests'],
            metrics=metrics, acceptance=receipt, checkpoint=checkpoint_receipt,
            timings=dict(collect_seconds=collect_seconds, update_seconds=update_seconds,
                         verify_seconds=verify_seconds))
        collective('record', lambda: record(report))
        if evaluate is not None:
            collective('evaluate', lambda: evaluate(session, report))
    return dict(completed_updates=session.completed_updates, policy_id=session.policy_id,
        schedule_sha256=schedule.sha256, phase=session.phase)
