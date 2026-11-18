"""Audit the committed learning prefix of an interrupted formal attempt.

Reuse the frozen per-update auditor without changing any live dependencies.
The caller must independently bind the package, terminal containers and complete
transport. Later, possibly incomplete output is not credited as learning history.
"""
import inspect
from pathlib import Path

from multimodal_jev.baseline import file_hash
from tools.reports import audit_context_macro_training as shared
from tools.reports.context_macro_formal_candidate import STEPS

AUDITOR_SHA256 = 'cd26e79996b01440ab26cd52ffbb0e9eafb8045928f697ec5805a4a489dc69bc'


def prefix_inventory(run, stop, checkpoint_steps):
    """Require every declared artifact up to stop; ignore the uncredited suffix."""
    run = Path(run)
    for directory, pattern, kind, expected_steps in [
        (run, 'policy-*-committed.json', 'policy', range(1, stop + 1)),
        (run, 'checkpoint-*-committed.json', 'checkpoint', checkpoint_steps),
        *[(run/f'rank-{rank}', 'checkpoint-*.pt', 'shard', checkpoint_steps)
          for rank in range(2)],
    ]:
        expected = {
            f'checkpoint-{step:04d}.pt' if kind == 'shard'
            else f'{kind}-{step:04d}-committed.json'
            for step in expected_steps
        }
        actual = set()
        for path in directory.glob(pattern):
            try:
                step = int(path.name.split('-')[1].split('.')[0])
            except (ValueError, IndexError) as exc:
                raise ValueError('invalid formal prefix artifact name') from exc
            if step <= stop:
                actual.add(path.name)
        if actual != expected:
            raise ValueError('complete exact committed prefix inventory required')


def prefix_journals(journal, count):
    actual = {p.name for p in journal.glob('batch-*')}
    expected = {f'batch-{i:08d}' for i in range(count)}
    if not expected <= actual:
        raise ValueError('missing consumed prefix macro journal')
    # A suffix may be incomplete after a crash; it is explicitly uncredited.
    for name in actual - expected:
        suffix = name.removeprefix('batch-')
        if not suffix.isdigit() or len(suffix) != 8 or int(suffix) < count:
            raise ValueError('invalid unconsumed macro journal name')


def audit_prefix(package, collection, stop):
    if type(stop) is not int or stop not in STEPS[1:-1]:
        raise ValueError('nonterminal sparse recovery boundary required')
    if file_hash(Path(shared.__file__)) != AUDITOR_SHA256:
        raise ValueError('frozen shared prefix auditor changed')
    source = inspect.getsource(shared.audit_outputs)
    replacements = [
        ("    stop=c['stop_update']", "    stop=prefix_stop"),
        ("    if start not in (0,1):raise ValueError('initial prefix or one-step replay required')",
         "    if start != 0:raise ValueError('fresh formal parent required for prefix audit')"),
        ("    if read(run/'result.json')!=dict(passed=True,start_update=start,stop_update=stop,schedule_updates=634,\n"
         "            learning_population=40513,engineering_prefix=not formal):raise ValueError('complete configured result required')",
         "    # An interrupted parent need not have a successful final result."),
        ("read(collection/f'service-{i}/terminal.json')['exit_code']!=0\n                or ", ""),
        ("checkpoint_steps=formal_plan['recovery_checkpoint_steps'] if formal else list(range(start+1,stop+1))",
         "checkpoint_steps=[s for s in formal_plan['recovery_checkpoint_steps'] if s<=stop]"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('unique prefix audit anchor required: ' + before)
        source = source.replace(before, after)
    begin = '    if formal:\n        for directory,pattern in '
    end = "    journal=collection/'journal'/client['attempt_id']"
    left, middle = source.split(begin)
    _, right = middle.split(end)
    source = left + '    prefix_inventory(run,stop,checkpoint_steps)\n' + end + right
    before = "    if {p.name for p in journal.glob('batch-*')}!={f'batch-{i:08d}' for i in range(sequence)}:raise ValueError('unaccounted macro batch')"
    if source.count(before) != 1:
        raise ValueError('unique prefix journal boundary required')
    source = source.replace(before, '    prefix_journals(journal,sequence)')
    begin = '    for rank in range(2):\n        r=read(run/f\'rank-{rank}/result.json\')'
    end = '    if start:\n        from tools.reports.compare_context_dual_throughput_pair import equal'
    left, middle = source.split(begin)
    _, right = middle.split(end)
    source = left + end + right
    namespace = dict(vars(shared), prefix_stop=stop,
                     prefix_inventory=prefix_inventory, prefix_journals=prefix_journals)
    exec(compile(source, '<formal-prefix-output-auditor>', 'exec'), namespace)
    result = namespace['audit_outputs'](package, collection, formal=True)
    result.update(status='formal_dp2_prefix_outputs_audited', start_update=0, stop_update=stop,
                  ignored_suffix=True, fresh_process_replay_verified=False,
                  scope='Committed formal prefix only; every consumed update, target, credit, '
                        'journal, export and activation checked. Master/Adam checked at sparse '
                        'boundaries. Package, transport, runtime and resumed segment are separate.')
    return result
