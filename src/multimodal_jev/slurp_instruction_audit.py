"""Independent schedule/cursor checks for terminal instruction checkpoints."""
import math
from pathlib import Path
from .slurp_instruction_resume import schedule_cursor


def completed_range(config, epoch_updates):
    from .simple_batch_config import batch_layout
    batch_layout(config)
    start = config.get('start_update', 0)
    count = config['updates']
    if (type(start) is not int or type(count) is not int or start < 0 or count < 1
            or start + count > sum(epoch_updates)):
        raise ValueError('bounded complete instruction update range required')
    interval = config.get('checkpoint_interval', 1)
    if type(interval) is not int or interval < 1:
        raise ValueError('positive checkpoint interval required')
    if start:
        if (not config.get('initial_adapter') or config['total_updates'] != sum(epoch_updates)
                or start + count != config['total_updates']):
            raise ValueError('terminal continuation must complete the full schedule')
    elif count != 2 or config.get('initial_adapter'):
        raise ValueError('fresh instruction audit requires the two-update prefix')
    return start, start + count, schedule_cursor(start + count, epoch_updates)


def exported_at(config, completed, epoch_updates):
    """Independently derive export boundaries from the frozen run configuration."""
    start, end, _ = completed_range(config, epoch_updates)
    return (start < completed <= end and
            (completed % config.get('checkpoint_interval', 1) == 0
             or schedule_cursor(completed, epoch_updates)[1] == 0 or completed == end))


def input_policy_id(config, root, step, epoch_updates, identity):
    """Match the exact immutable training lineage, including non-export updates.

    Non-export identifiers are cursor labels, not independent weight hashes.
    The terminal master/export audit establishes the actual endpoint identity.
    """
    start, end, _ = completed_range(config, epoch_updates)
    if not start <= step < end:
        raise ValueError('batch outside executed update range')
    run = Path(root) / 'run'
    initial = Path(config['initial_adapter']) if start else run/'adapter-step0000'
    if step == start:
        adapter_id = identity(initial)
    elif exported_at(config, step, epoch_updates):
        adapter_id = identity(run/f'adapter-step{step:04d}')
    else:
        adapter_id = identity(initial) + f':instruction-update{step}'
    return config['base_id'] + ':' + adapter_id


def verify_rank_report(report, terminal, config, *, rank, epoch_updates, adapter_id):
    start, end, _ = completed_range(config, epoch_updates)
    if (terminal.get('exit_code') != 0 or report.get('passed') is not True
            or report.get('rank') != rank or report.get('updates') != end
            or report.get('executed_updates') != end-start
            or report.get('base_id') != config['base_id']
            or report.get('final_adapter_id') != adapter_id):
        raise ValueError('successful exact rank endpoint required')
    records = report['step_metrics']
    if [r['update'] for r in records] != list(range(start+1, end+1)):
        raise ValueError('complete ordered per-rank update ledger required')
    for record in records:
        norms = record['metrics']['actor/grad_norm']
        if (len(norms) != 1 or not math.isfinite(norms[0]) or norms[0] <= 0
                or not math.isfinite(record['seconds']) or record['seconds'] < 0):
            raise ValueError('finite nonzero update evidence required')
    dtypes = report.get('execution_dtypes', {})
    if len(dtypes) != 112 or any(v != ['torch.bfloat16', 'torch.bfloat16'] for v in dtypes.values()):
        raise ValueError('all112 BF16 LoRA execution hooks required')
    return len(records)
