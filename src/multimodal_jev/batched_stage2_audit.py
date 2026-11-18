"""Independent Stage2 source/process and stage-boundary evidence checks."""
import json
from pathlib import Path

from .baseline import file_hash
from .resident_service_admission import IMAGE
from .batched_stage2_initialization import authenticate


def compare_initial_state(source, initial, *, rank, reference_policy_id, schedule_sha256):
    """Compare real FP32 masters against Stage1; require empty new Adam state."""
    import torch
    expected = dict(rank=rank, world_size=4, updates=0, next_update=0,
        stage='simple_stage2_initial', reference_policy_id=reference_policy_id,
        schedule_sha256=schedule_sha256, adapter_id=reference_policy_id.split(':', 1)[1])
    if (type(rank) is not int or rank not in range(4)
            or source.get('rank') != rank or source.get('world_size') != 4 or source.get('updates') != 634
            or any(initial.get(k) != v for k, v in expected.items())
            or len(source['names']) != 112 or len(set(source['names'])) != 112
            or initial['names'] != source['names']
            or set(initial['parameters']) != set(source['names'])
            or set(source['parameters']) != set(source['names'])
            or initial['optimizer']['param_groups'] != source['optimizer']['param_groups']
            or initial['optimizer']['state']):
        raise ValueError('exact final Stage1 layout and empty Stage2 optimizer required')
    for name in source['names']:
        left, right = source['parameters'][name], initial['parameters'][name]
        if (left.dtype != torch.float32 or right.dtype != torch.float32 or left.shape != right.shape
                or not torch.isfinite(left).all() or not torch.isfinite(right).all()
                or not torch.equal(left, right)):
            raise ValueError('Stage2 initial master differs from final Stage1')
    return dict(passed=True, rank=rank, exact_masters=True, optimizer_reset=True, tensors=112)


def audit_boundary(config):
    """An explicit stopped prefix is evidence for recovery, not a finished run."""
    mode = config.get('audit_mode', 'terminal')
    if mode == 'terminal':
        if 'completed_updates' in config:
            raise ValueError('terminal audit cannot override the full budget')
        return 634, False
    updates = config.get('completed_updates')
    if (mode != 'stopped_prefix' or type(updates) is not int or not 1 <= updates < 634):
        raise ValueError('explicit nonterminal stopped-prefix cursor required')
    return updates, True


def source(config, *, verify_parity):
    """Authenticate a terminal run or a stopped prefix and saved initial states.

    A crash may leave no terminal.json. Prefix mode still requires an exact
    pinned exited container; it never infers termination from missing files.
    """
    import torch
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only independent Stage2 audit required')
    boundary, prefix = audit_boundary(config)
    files = {}
    def read(path):
        path = Path(path); files[str(path)] = file_hash(path)
        return json.loads(path.read_text())
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('Stage2 audit dependency changed')
        files[path] = digest
    root = Path(config['training_root'])
    train = read(root / 'config.json'); launch = read(config['training_launch'])
    observed = read(config['training_observed'])
    if (config['pins'].get(str(root / 'config.json')) != files[str(root / 'config.json')]
            or config['pins'].get(config['training_launch']) != files[config['training_launch']]
            or config['pins'].get(config['training_observed']) != files[config['training_observed']]
            or launch['image'] != IMAGE or observed['image'] != IMAGE
            or launch['container_id'] != observed['container_id']
            or observed['state']['Status'] != 'exited'
            or type(observed['state'].get('ExitCode')) is not int
            or (not prefix and observed['state']['ExitCode'] != 0)
            or launch['config_sha256'] != file_hash(root / 'config.json')
            or 'sampled_probe' in train):
        raise ValueError('pinned exited formal Stage2 process required')
    if not prefix and read(root / 'terminal.json')['exit_code'] != 0:
        raise ValueError('successful formal Stage2 terminal required')
    recovered = None
    if 'continuous_resume' in train:
        from .batched_stage2_resume import authenticate as resume
        recovered = resume(train, verify_parity=verify_parity)
        if recovered['completed_updates'] >= boundary:
            raise ValueError('resumed audit must cover at least one new update')
        initialized = recovered['initial']
        files.update(recovered['files'])
    else:
        initialized = authenticate(train, verify_parity=verify_parity)
    files.update(initialized['files'])
    from .span_batch_training_config import batch_layout
    layout = batch_layout(train)
    plan = train['stage2_initialization']
    from .simple_resident_selection import SELECTION_UPDATES
    if (type(plan.get('stop_update')) is not int or plan['stop_update'] != 634
            or plan.get('selection_updates') != list(SELECTION_UPDATES)):
        raise ValueError('complete frozen Stage2 budget and selection schedule required')
    checks = []; names = None
    for rank in range(4):
        out = root / 'run' / f'rank-{rank}'
        if not prefix and read(out / 'terminal.json')['exit_code'] != 0:
            raise ValueError('all Stage2 ranks must have succeeded')
        execution_path = (root / 'run' / f'loop-rank-{rank}' / f'execution-{boundary:04d}.json'
                          if prefix else out / 'stage2-execution.json')
        execution_record = read(execution_path)
        if prefix and (execution_record.get('updates') != boundary
                       or execution_record.get('rank') != rank):
            raise ValueError('checkpoint-specific BF16 execution evidence required')
        execution = execution_record['execution_dtypes']
        if len(execution) != 112 or any(v != ['torch.bfloat16', 'torch.bfloat16'] for v in execution.values()):
            raise ValueError('complete BF16 Stage2 execution hooks required')
        saved = out / ('stage2-resumed.pt' if recovered else 'stage2-initial.pt')
        files[str(saved)] = file_hash(saved)
        actual = torch.load(saved, map_location='cpu', weights_only=True)
        if (actual.get('candidate_profile') != train['candidate_profile']
                or actual.get('batch_layout') != layout or actual.get('batch_order') != train['batch_order']):
            raise ValueError('candidate initial/restored execution contract differs')
        if recovered:
            from .intent_resident_replay import compare_state
            checks.append(compare_state(recovered['states'][rank], actual, rank=rank,
                updates=recovered['completed_updates'],continuous=True,formal=True))
        else:
            before = torch.load(initialized['shard_paths'][rank], map_location='cpu', weights_only=True)
            checks.append(compare_initial_state(before, actual, rank=rank,
                reference_policy_id=initialized['reference_policy_id'], schedule_sha256=train['schedule_sha256']))
        if names is not None and names != actual['names']:
            raise ValueError('initial cross-rank parameter order differs')
        names = actual['names']
    return dict(files=files, initialization=initialized, checks=checks, names=names,
                completed_updates=boundary, stopped_prefix=prefix, recovery=recovered)


def selections(root, train, policies, files, *, through_update=634, history=()):
    """Recompute all preregistered full-selection points and rank bindings."""
    from .simple_resident_selection import SELECTION_UPDATES, verify_result
    from .slurp_official_training import load
    rows, task, _ = load(train['release'], train['release_sha256'], 'selection', purpose='evaluate')
    if type(through_update) is not int or not 1 <= through_update <= 634:
        raise ValueError('bounded selection audit cursor required')
    root = Path(root); results = []
    steps = tuple(step for step in SELECTION_UPDATES if step <= through_update)
    prior = {item['updates']: item for item in history}
    if len(prior) != len(history) or list(prior) != list(steps[:len(history)]):
        raise ValueError('ordered selection history prefix required')
    expected = {f'selection-{step:04d}' for step in steps if step not in prior}
    directory = root / 'run/evaluation'
    actual = {p.name for p in directory.iterdir()} if directory.exists() else set()
    # A failed suffix can have partially written future evaluations. Do not
    # accept them, delete them, or require them for an earlier recovery point.
    allowed_suffix = {f'selection-{step:04d}' for step in SELECTION_UPDATES if step > through_update}
    if not expected <= actual or actual - expected - allowed_suffix:
        raise ValueError('exact preregistered selection directories required')
    for step in steps:
        path = Path(prior[step]['result']) if step in prior else root / 'run/evaluation' / f'selection-{step:04d}/result.json'
        digest = file_hash(path)
        if step in prior and (prior[step]['result_sha256']!=digest or prior[step]['policy_id']!=policies[step]):
            raise ValueError('inherited selection identity differs')
        report = verify_result(path, digest, rows, task, release_sha256=train['release_sha256'])
        if report['updates'] != step or report['policy_id'] != policies[step]:
            raise ValueError('selection point belongs to another trained checkpoint')
        raw = json.loads(path.read_text()); files[str(path)] = digest
        files.update(raw['files']); files.update(raw['checkpoint']['files'])
        files[raw['inputs']] = raw['inputs_sha256']
        for rank in range(4):
            source_root = path.parents[3] if step in prior else root
            binding = source_root / 'run' / f'loop-rank-{rank}' / f'selection-{step:04d}.json'
            value = json.loads(binding.read_text()); files[str(binding)] = file_hash(binding)
            if (value.get('passed') is not True or value['updates'] != step
                    or value['policy_id'] != policies[step] or value['recordings'] != 9226
                    or value['result'] != str(path) or value['result_sha256'] != digest
                    or value['metrics'] != {k: report['metrics'][k] for k in
                        ('accuracy', 'macro_F1_fixed60', 'macro_recall_supported', 'invalid_rate')}):
                raise ValueError('rank selection binding differs from independent metrics')
        results.append(dict(report,result=str(path)))
    return results
