"""Deterministic, read-only status snapshot for the simple-task experiment ladder.

Reads only local immutable records: driver start files, stage terminals and
independent audit summaries. No SSH, Docker, model call, dispatch, restart,
cleanup or configuration change; the only write is the snapshot itself. The
script is safe to call repeatedly and makes no remote observation, so it does
not consume the 900-second ordinary remote-observation interval.
"""
import argparse
import fcntl
import glob
import json
import os
from pathlib import Path
import time

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def encode(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())


def replace(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '-' + str(os.getpid()) + '.tmp')
    with temporary.open('x') as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def start_ticks(pid):
    try:
        return Path('/proc/' + str(pid) + '/stat').read_text().split(') ', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def cmdline(pid):
    try:
        raw = Path('/proc/' + str(pid) + '/cmdline').read_bytes()
    except OSError:
        return None
    return raw.decode().rstrip('\0').split('\0')


def driver_state(path):
    """Exact recorded-process identity; a reused pid is not a running driver."""
    record = read(path)
    if record is None:
        return dict(state='no_driver_record')
    pid = record.get('pid')
    if type(pid) is not int or not isinstance(record.get('process_start_ticks'), str):
        raise ValueError('driver record lacks pid or process start ticks')
    ticks = start_ticks(pid)
    if ticks is None:
        return dict(pid=pid, state='exited', started_unix=record.get('started_unix'))
    if ticks != record['process_start_ticks']:
        return dict(pid=pid, state='pid_reused_identity_differs', started_unix=record.get('started_unix'))
    return dict(pid=pid, state='running', started_unix=record.get('started_unix'),
                elapsed_seconds=round(time.time() - record['started_unix'], 1),
                argv_matches=cmdline(pid) == record.get('argv'))


DEFAULT_MARKERS = (('audit.json', 'audit_without_terminal'),
                   ('audit-config.json', 'collection_or_independent_audit'),
                   ('monitor-config.json', 'training_monitored'),
                   ('results.json', 'transfer_scored'),
                   ('grpo-source-plan.json', 'grpo_source_planned'),
                   ('deployment.json', 'dispatched'),
                   ('source-admission.json', 'source_admitted'),
                   ('curve-config.json', 'curve_collecting'),
                   ('waiting.json', 'waiting_predecessor'),
                   ('retirement-complete.json', 'old_queue_retired'),
                   ('retirement-intent.json', 'retiring_old_queue'),
                   ('config.json', 'launching'),
                   ('preparation-config.json', 'prepared'),
                   ('template.json', 'template_written'),
                   ('provenance.json', 'configured'))


def phase_root(spec):
    """Directory whose markers describe the stage; a controller may keep its
    preparation records apart from the launched job's finalization directory."""
    if spec.get('root'):
        return Path(spec['root'])
    return Path(spec['terminal']).parent


def stage_phase(spec):
    terminal = read(spec['terminal']) if spec.get('terminal') else None
    if terminal is not None:
        return 'terminal', terminal
    root = phase_root(spec)
    progress = spec.get('progress')
    if progress and root.is_dir():
        done = len(glob.glob(progress['glob']))
        if done:
            return f"{done}_of_{progress['planned']}_{progress.get('label', 'stages')}", None
    for name, phase in spec.get('markers', DEFAULT_MARKERS):
        if (root / name).exists():
            return phase, None
    return ('absent' if not root.is_dir() else 'empty_directory'), None


def gate(spec):
    """Declared predecessor terminal of a waiting stage, with its current status."""
    root = phase_root(spec)
    waiting = read(root / 'waiting.json')
    if waiting is None:
        return None
    predecessor = waiting.get('predecessor') or waiting
    path = predecessor.get('terminal') if isinstance(predecessor, dict) else None
    if path is None:
        return None
    terminal = read(path)
    return dict(path=path, present=terminal is not None,
                status=None if terminal is None else terminal.get('status'))


def queue_events(directory):
    """Deterministic read of the controller's own dispatch event log."""
    files = sorted(Path(directory).glob('[0-9]*.json'))
    if not files:
        return None
    counts = {}
    for path in files:
        kind = read(path).get('event')
        counts[kind] = counts.get(kind, 0) + 1
    last = read(files[-1])
    return dict(count=len(files), counts=counts,
                last=dict(event=last.get('event'), tag=last.get('tag'),
                          unix=last.get('unix')))


def audited_reward(audit):
    """Per-update branch reward from an independent audit; never inferred here."""
    updates = (audit or {}).get('updates')
    if not isinstance(updates, list) or not updates:
        return None
    rows = []
    for update in updates:
        row = dict(update=update.get('update'), real_queries=update.get('real_queries'))
        for name in ('ranking', 'existence'):
            branch = (update.get('branches') or {}).get(name) or {}
            if 'mean_weighted_reward' in branch:
                row[name] = branch['mean_weighted_reward']
        rows.append(row)
    return rows


def provisional_reward(collection):
    """Same statistic as the independent auditor, from archived group tensors only.

    This is not the audit: it does not re-verify labels, wire identity, inputs or
    advantages. It is reported explicitly as unaudited so no number is mistaken
    for an independent result.
    """
    files = sorted(Path(collection).glob('groups-*.pt'))
    if not files:
        return None
    import torch
    torch.set_num_threads(1)
    rows = []
    for path in files:
        groups = torch.load(path, map_location='cpu', weights_only=False)
        real = sum(1 for group in groups if group['real_request'])
        if real < 1:
            raise ValueError('provisional reward requires real requests')
        row = dict(update=int(path.stem.split('-')[-1]), real_queries=real)
        for name in ('ranking', 'existence'):
            total = 0.0
            for group in groups:
                if not group['real_request']:
                    continue
                branch = group['dual_credit']['branches'][name]
                total += float((branch['weight'] * branch['rewards'].mean(1)).sum())
            row[name] = total / real
        rows.append(row)
    return rows


DEV_KEYS = ('query_macro_set_f1', 'top1_event_accuracy', 'candidate_binary_nll',
            'candidate_brier', 'exact_set', 'mean_selected_count')


def dev_point(finalization):
    audit = read(Path(finalization) / 'audit.json')
    if audit is None:
        return None
    metrics = (audit.get('result') or {}).get('metrics') or {}
    return dict(step=audit.get('source_step'), status=audit.get('status'),
                metrics={key: metrics[key] for key in DEV_KEYS if key in metrics})


def summarize(spec):
    row = dict(name=spec['name'], kind=spec['kind'])
    phase, terminal = stage_phase(spec)
    row['phase'] = phase
    if terminal is not None:
        row.update(exit_code=terminal.get('exit_code'), status=terminal.get('status'),
                   accepted=terminal.get('status') in spec.get('accepted', []))
        for key in ('error', 'points', 'updates', 'real_exposures', 'reference_endpoint'):
            if key in terminal:
                row[key] = terminal[key]
    if spec.get('driver'):
        row['driver'] = driver_state(spec['driver'])
    if phase in ('waiting_predecessor', 'absent', 'empty_directory'):
        row['gate'] = gate(spec)
    if spec.get('events'):
        row['events'] = queue_events(spec['events'])
    if spec.get('retired'):
        row['retired'] = spec['retired']
    reward = spec.get('train_reward')
    if reward:
        audit = read(reward['audit']) if reward.get('audit') else None
        rows = audited_reward(audit)
        if rows is not None:
            row['train_reward'] = dict(source='independent_audit', audited=True, updates=rows)
        elif reward.get('collection') and reward.get('provisional'):
            try:
                rows = provisional_reward(reward['collection'])
            except (ImportError, FileNotFoundError):
                rows = None
            row['train_reward'] = (dict(source='pending', audited=False, updates=None,
                                        reason='independent audit not committed') if rows is None else
                                   dict(source='archived_group_tensors', audited=False, updates=rows,
                                        reason='aggregated from archived groups; independent audit pending'))
        else:
            row['train_reward'] = dict(source='pending', audited=False, updates=None,
                                       reason='independent audit not committed')
    if spec.get('reward_note'):
        row['reward_note'] = spec['reward_note']
    if spec.get('dev_points'):
        found = [dev_point(p) for p in sorted(glob.glob(spec['dev_points']))]
        found = [p for p in found if p is not None]
        row['dev_points'] = found
        row['dev_points_planned'] = spec.get('dev_points_planned')
    return row


def stage_line(row):
    head = f"{row['name']:<22} {row['phase']:<30}"
    if 'status' in row:
        head += f" {row['status']} (exit {row['exit_code']})"
    driver = row.get('driver') or {}
    if driver.get('state') == 'running':
        head += f" [pid {driver['pid']} {driver['elapsed_seconds'] / 3600:.2f}h]"
    elif driver.get('state') not in (None, 'no_driver_record'):
        head += f" [driver {driver['state']}]"
    events = row.get('events')
    if events:
        head += (f" [事件 {events['count']} 条，末条 {events['last']['event']}"
                 f" {events['last']['tag'] or ''}]")
    if row.get('retired'):
        head += f" [已被异步调度取代：{row['retired']}]"
    return head


def reward_line(row):
    reward = row.get('train_reward')
    if not reward:
        return None
    if not reward.get('updates'):
        if row.get('phase') in ('absent', 'empty_directory'):
            return None
        return f"  {row['name']}: 不可用 —— {reward.get('reason')}"
    tag = '' if reward['audited'] else '  [provisional，独立审核未完成]'
    rows = [u for u in reward['updates'] if 'ranking' in u and 'existence' in u]
    if not rows:
        return f"  {row['name']}: 不可用 —— 审核未记录分支配对 reward{tag}"
    if len(rows) <= 4:
        body = '; '.join(
            f"update {u['update']}: ranking {u['ranking']:.6f} / existence {u['existence']:.6f}"
            for u in rows)
    else:
        mean_ranking = sum(u['ranking'] for u in rows) / len(rows)
        mean_existence = sum(u['existence'] for u in rows) / len(rows)
        body = (f"{len(rows)} 个 update 平均 ranking {mean_ranking:.6f} / existence {mean_existence:.6f}"
                f"（首 update {rows[0]['update']}: {rows[0]['ranking']:.6f} / {rows[0]['existence']:.6f}；"
                f"末 update {rows[-1]['update']}: {rows[-1]['ranking']:.6f} / {rows[-1]['existence']:.6f}）")
    queries = sorted({u.get('real_queries') for u in rows if u.get('real_queries') is not None})
    scope = f"  [每条 update real_queries={queries}]" if len(queries) == 1 else ''
    return f"  {row['name']}: {body}{scope}{tag}"


def render(record):
    lines = [f"实验状态快照 {record['observed_utc']}",
             "来源：本地冻结产物确定性读取；无 SSH/Docker/模型调用/派发/重启。", "", "链路："]
    lines += ['  ' + stage_line(row) for row in record['stages']]
    notes = [f"  {row['name']}: {row['reward_note']}" for row in record['stages'] if row.get('reward_note')]
    if notes:
        lines += ['', '说明：'] + notes
    trains = [line for line in (reward_line(row) for row in record['stages']) if line]
    lines += ['', 'TRAIN reward（dual-credit GRPO：ranking=标签冲突候选对的加权平均，existence=逐候选类平衡加权平均）']
    lines += trains or ['  尚无已审核 reward。']
    idle = [row['name'] for row in record['stages']
            if row.get('train_reward') and row.get('phase') in ('absent', 'empty_directory')]
    if idle:
        lines.append('  未开始（无派发记录，尚无 reward）：' + '、'.join(idle))
    for row in record['stages']:
        if row.get('dev_points_planned'):
            points = row.get('dev_points') or []
            lines += ['', f"DEV/selection（{row['name']}，9226 条，不参与训练、奖励或调参）"]
            if points:
                for point in points:
                    values = point['metrics']
                    shown = ' / '.join(f'{key} {values[key]:.6f}' for key in
                                       ('query_macro_set_f1', 'top1_event_accuracy')
                                       if isinstance(values.get(key), float))
                    lines.append(f"  step {point['step']}: {shown}")
            else:
                lines.append(f"  不可用 —— 已完成 {len(points)}/{row['dev_points_planned']} 个预登记选点；"
                             f"阶段 {row['phase']}，前驱门禁：{(row.get('gate') or {}).get('status') or '未产生'}")
    return '\n'.join(lines)


def snapshot(config):
    stamp = time.time()
    record = dict(schema='context-experiment-status-v1', observed_unix=stamp,
                  observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(stamp)),
                  scope='Local immutable artifacts only; no remote observation, dispatch or model call.',
                  stages=[summarize(spec) for spec in config['stages']])
    record['ladder_complete'] = all(row.get('accepted') for row in record['stages']
                                    if not row.get('retired'))
    return record


def run(template, local):
    config, provenance = load_config(template, local)
    if config.get('schema') != 'context-experiment-status-v1':
        raise ValueError('unsupported status configuration schema')
    output = Path(config['output'])
    check_paths({'status': output})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest = read(output / 'latest.json')
        stamp = time.time()
        if latest is not None and stamp < latest['observed_unix'] + config['interval_seconds']:
            return dict(status='not_due_no_snapshot',
                        next_snapshot_unix=latest['observed_unix'] + config['interval_seconds'])
        record = snapshot(config)
        record['config_provenance'] = provenance
        name = 'snapshot-' + str(time.time_ns()) + '.json'
        write_new(output / name, record)
        replace(output / 'latest.json', record)
        return dict(status='snapshot_written', path=str(output / name), record=record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = run(args.config, args.local)
    if args.json or result['status'] != 'snapshot_written':
        print(json.dumps({k: v for k, v in result.items() if k != 'record'}, indent=2))
        return
    print(render(result['record']))
    print('\n快照：' + result['path'])


if __name__ == '__main__':
    main()
