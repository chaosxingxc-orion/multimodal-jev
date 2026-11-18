"""CPU differential between the composed controller and the frozen rewrite.

The frozen controller (``context_async_pipeline`` and its shared audit
successor) reaches behaviour by editing source text at run time. This audit
drives that rewrite and the composed module on identical frozen inputs and
writes one deterministic verdict.

Comparison basis, stated rather than assumed: the v0.0.49 controller died
before publishing terminal products (``driver.log`` is empty, no
``terminal.json`` exists, ``infonce/`` holds only ``template.json``), so a
field-by-field product comparison against a completed run is not available.
The differential is therefore re-based onto frozen inputs that exist on disk:

1. Docker argv from the frozen mount builder and its two-anchor rewrite,
   compared with the composed builder over every real prepared package, in
   both producer modes.
2. The launch lifecycle: both wrappers run under one recording effect set, and
   their external call sequence, written artifacts, queue events and return
   value are compared. The frozen side is the controller's own rewrite; its
   effect globals are replaced in place on the same code object. The real
   ``check_runtime`` proof runs on both sides.
3. The selection curve: the frozen rewritten function and the composed
   function run with identical collaborators; published files, the call and
   queue multiset (the curve fans out over a thread pool, so order is not fixed
   by either side) and the returned curve are compared after root normalisation.
4. The admission chain: the composed entry must return the frozen result
   unchanged, through the frozen function.
5. Seam bindings: same ``__code__`` objects and the same injected globals as
   the frozen ``optimized_selection`` builds.

Not covered here, and not claimed: NPU execution, ssh/docker/scp transport,
collection transfer, and every downstream numerical audit. Dispatch is refused
by the composed module this round. The audit never calls a model, never
contacts a host and never writes to the queue.

The verdict is deterministic: a non-empty ``failures`` list is a rejected
report, not a warning.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import collect_context_replica_infonce_curve as curve_module
from tools.reports import context_async_pipeline as frozen
from tools.reports import context_audit_optimized_pipeline as admission_engine
from tools.reports import context_cached_storage_operations as storage
from tools.reports import context_composed_pipeline as composed
from tools.reports import context_replica_training_operations as shared
from tools.reports.context_dual_probe_operations import check_runtime
from tools.reports.prepare_context_dual_throughput_pair import sha

SCHEMA = 'context-composed-pipeline-differential-v1'
# Constructs the composed path must not use. FunctionType is counted apart: on
# an unmodified code object it is checked injection, not a source rewrite.
REWRITE_NAMES = ('exec', 'eval')
REWRITE_ATTRIBUTES = ('getsource', 'getsourcelines')
INJECTION_NAMES = ('FunctionType',)
CONTROLLER_PATH = ('context_async_pipeline.py', 'context_audit_optimized_pipeline.py',
                   'context_cached_storage_operations.py', 'context_replica_trainer_candidate.py',
                   'context_composed_pipeline.py')


def digest(payload):
    return hashlib.sha256(payload.encode()).hexdigest()


def normalise(value, root):
    """Replace one run root so two sides compare on structure, not paths."""
    if isinstance(value, str):
        return value.replace(str(root), '<root>')
    if isinstance(value, dict):
        return {key: normalise(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(item, root) for item in value]
    return value


def rewriting_sites(path):
    """Run-time source rewriting and injection sites in one module."""
    path = Path(path)
    tree = ast.parse(path.read_text())
    found, injection = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None
        attribute = func.attr if isinstance(func, ast.Attribute) else None
        if name in REWRITE_NAMES:
            found.append(f'{path.name}:{node.lineno}:{name}')
        elif attribute in REWRITE_ATTRIBUTES:
            found.append(f'{path.name}:{node.lineno}:{attribute}')
        elif attribute in INJECTION_NAMES or name in INJECTION_NAMES:
            injection.append(f'{path.name}:{node.lineno}:{attribute or name}')
    return found, injection


def repository_debt(root):
    """Rewrite and injection sites across the measured source tree."""
    rows, sites, injections = [], [], []
    for directory in ('src', 'tools'):
        for path in sorted((Path(root)/directory).rglob('*.py')):
            found, injection = rewriting_sites(path)
            if not found and not injection:
                continue
            sites.extend(found)
            injections.extend(injection)
            rows.append(dict(file=str(path.relative_to(root)), rewrite_sites=len(found),
                             injection_sites=len(injection),
                             controller_path=path.name in CONTROLLER_PATH, sites=found))
    return dict(files=len(rows), rewrite_sites=len(sites), injection_sites=len(injections),
                controller_path_files=[row['file'] for row in rows if row['controller_path']],
                rows=rows, sites=sites)


class Recorder:
    """One effect set that never touches a host, a queue or an NPU."""

    def __init__(self, image):
        self.image = image
        self.calls = []
        self.argv = None

    @staticmethod
    def result(stdout, returncode=0):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr='')

    def run(self, argv, **kwargs):
        self.calls.append(dict(kind='run', argv=[str(value) for value in argv]))
        answer = list(argv)
        if answer[0] == 'ssh':
            command = answer[-1]
            if 'docker image inspect' in command:
                return self.result(json.dumps([{'Id': self.image}]))
            if 'npu-smi info' in command:
                return self.result('\n'.join(f'No running processes found in NPU {device}'
                                             for device in (4, 5, 6, 7)))
            if 'docker create' in command or 'docker start' in command:
                return self.result('a'*64)
            if 'docker inspect' in command:
                return self.result(json.dumps([self.inspect()]))
            return self.result('')
        if answer[0] == 'scp':
            return self.result('')
        raise AssertionError('unexpected process call: ' + repr(answer[:2]))

    def inspect(self):
        """A docker inspect consistent with the argv actually dispatched."""
        argv = self.argv
        values = lambda flag: [argv[i+1] for i, value in enumerate(argv) if value == flag]
        return {'Id': 'a'*64, 'Image': self.image,
                'State': {'Status': 'exited', 'ExitCode': 0, 'OOMKilled': False},
                'HostConfig': {
                    'Privileged': False, 'NetworkMode': 'none', 'Binds': values('-v'),
                    'Devices': [dict(zip(('PathOnHost', 'PathInContainer',
                                         'CgroupPermissions'), (*spec.split(':'), 'rwm')))
                                for spec in values('--device')]},
                'Config': {'Entrypoint': values('--entrypoint'),
                           'Cmd': argv[argv.index(self.image)+1:],
                           'Env': values('-e')}}

    @contextmanager
    def taropen(self, path, mode):
        self.calls.append(dict(kind='tar.open', name=str(Path(path).name), mode=mode))
        Path(path).write_bytes(b'archive')
        yield SimpleNamespace(add=lambda *args, **kwargs: None)

    def check_paths(self, paths):
        self.calls.append(dict(kind='check_paths', paths=sorted(paths)))
        return []

    def write(self, path, value):
        self.calls.append(dict(kind='write', name=str(Path(path).name), value=value))
        with Path(path).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')

    def collect(self, record, output, host='npu'):
        self.calls.append(dict(kind='collect', manifest_sha256=record['manifest_sha256'],
                               output=str(Path(output).name), host=host))
        return {'status': 'transport_complete_independent_audit_required'}

    def wait(self, remote, container):
        self.calls.append(dict(kind='wait_exact', container_id=container))
        return 0

    def namespace(self):
        """Names as the frozen lifecycle resolves them, and as ``EFFECTS``."""
        import fcntl
        return dict(subprocess=SimpleNamespace(run=self.run),
                    tarfile=SimpleNamespace(open=self.taropen), fcntl=fcntl,
                    check_paths=self.check_paths, write=self.write, sha=sha,
                    collect=self.collect, check_runtime=check_runtime, wait_exact=self.wait)


class Queue:
    """The ready-queue seam, recorded rather than entered."""

    def __init__(self):
        self.events = []
        self.failed = False

    def check(self):
        self.events.append('check')
        if self.failed:
            raise RuntimeError('pipeline failure: no further dispatch')

    def abort(self):
        self.failed = True
        self.events.append('abort')

    @contextmanager
    def slot(self, priority, tag):
        self.events.append(f'enter:{priority}:{tag}')
        try:
            yield
        finally:
            self.events.append('exit')


def stage_inputs(package, deployment, tag, evidence):
    """The frozen package's own manifest and configuration.

    Admission is not re-run here: it hashes 400 MB of prepared packages and is
    compared separately in :func:`admission_case`."""
    root = Path(package)
    m = json.loads((root/'PREPARED.json').read_text())
    config = json.loads((root/'config.json').read_text())
    return m, config, dict(deployment, release_evidence=evidence, package=str(root),
                           manifest_sha256='0'*64, tag=tag, host='npu')


def evidence_root(root, config, destination):
    """Byte-identical copies of the release pins the frozen verifier needs.

    The frozen release proves its publication pins at absolute host paths; a
    declared relocation root supplies copies. Nothing outside ``root`` is
    written. The reference carries exactly the two names the frozen
    ``release_evidence`` accepts; the copy count is reported separately.
    """
    root = Path(root)
    publication = json.loads((Path(config['release'])/'publication-config.json').read_text())
    copied = 0
    for name, value in publication['pins'].items():
        if not Path(name).is_relative_to(destination):
            continue
        target = root/Path(name).relative_to(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
        if file_hash(target) != value:
            raise ValueError('relocated release evidence differs: ' + name)
        copied += 1
    if not copied:
        raise ValueError('release publishes no pins under the evidence destination')
    return dict(root=str(root), destination=destination), copied


def argv_cases(config, evidence):
    """Frozen argv versus composed argv over every real prepared package."""
    prior = {name: json.loads(Path(value['path']).read_text())
             for name, value in config['sources'].items()}
    packages = [('stage1_endpoint', config['preprepared_stage1_endpoint']['package']),
                ('finite_prefix', prior['finite']['stages']['prefix']['package']),
                ('finite_tail', prior['finite']['stages']['tail']['package'])]
    cases = []
    for name, package in packages:
        if not Path(package).exists():
            cases.append(dict(name=name, package=package, exists=False, identical=False))
            continue
        m, stage, c = stage_inputs(package, config['deployment'],
                                   'jev-differential-' + name.replace('_', '-'), evidence)
        for producer in (True, False):
            try:
                expected, expected_error = storage.command(c, m, stage, producer=producer), None
            except Exception as error:                       # noqa: BLE001 - recorded
                expected, expected_error = None, f'{type(error).__name__}: {error}'
            try:
                actual, actual_error = composed.command(c, m, stage, producer=producer), None
            except Exception as error:                       # noqa: BLE001 - recorded
                actual, actual_error = None, f'{type(error).__name__}: {error}'
            cases.append(dict(name=name, package=package, producer=producer, exists=True,
                              identical=(expected == actual
                                         and expected_error == actual_error),
                              frozen_error=expected_error, composed_error=actual_error,
                              tokens=len(actual) if actual else 0,
                              binds=len(storage.mounts(actual)) if actual else 0,
                              argv_sha256=digest(json.dumps(actual, sort_keys=True))
                              if actual else None))
    return cases


def lifecycle_case(name, package, config, work, producer, evidence):
    """Run both launch wrappers on one frozen package under recording effects.

    The frozen side is ``context_async_pipeline.launcher``'s own rewrite. Its
    effect names are replaced inside the very globals dict the exec'd function
    uses, so the code object, control flow and injected seams stay the
    controller's; only the external world is recorded.
    """
    m, stage, c = stage_inputs(package, config['deployment'], 'jev-differential-' + name,
                               evidence)
    validate = lambda _c: (m, stage, {})                  # noqa: E731 - injected seam
    authorization = ('selection_dispatch_allowed' if producer
                     else 'finite_training_dispatch_allowed')
    sides = {}
    for side in ('frozen', 'composed'):
        root = Path(work)/f'lifecycle-{name}-{side}'
        if root.exists():
            shutil.rmtree(root)
        (root/'ops').mkdir(parents=True)
        (root/'lease.lock').write_text('lease\n')
        run_config = dict(c, operations_root=str(root/'ops'), lease=str(root/'lease.lock'))
        recorder = Recorder(shared.IMAGE)
        queue = Queue()
        recorder.argv = composed.command(run_config, m, stage, producer=producer)
        if side == 'frozen':
            wrapper = frozen.launcher(SimpleNamespace(validate=validate), queue, 0,
                                      producer=producer, authorization=authorization)
            closure = dict(zip(wrapper.__code__.co_freevars,
                               (cell.cell_contents for cell in wrapper.__closure__)))
            closure['scope'].update(recorder.namespace())
            launch = wrapper
        else:
            launch = composed.stage_launcher(validate, queue, 0, producer=producer,
                                             authorization=authorization,
                                             effects=SimpleNamespace(**recorder.namespace()))
        try:
            result = launch(run_config, {'case': name})
        except Exception as error:                        # noqa: BLE001 - recorded
            result = dict(error=f'{type(error).__name__}: {error}')
        sides[side] = dict(calls=normalise(recorder.calls, root), result=normalise(result, root),
                           queue=normalise(queue.events, root), failed=queue.failed)
    same = {key: sides['frozen'][key] == sides['composed'][key]
            for key in ('calls', 'result', 'queue', 'failed')}
    return dict(name=name, package=str(package), producer=producer,
                identical=all(same.values()), differences=same,
                calls=len(sides['composed']['calls']),
                artifacts=[call['name'] for call in sides['composed']['calls']
                           if call['kind'] == 'write'],
                exec_argv_sha256=digest(json.dumps(
                    [call['argv'] for call in sides['composed']['calls']
                     if call['kind'] == 'run'], sort_keys=True)),
                result=sides['composed']['result'], queue=sides['composed']['queue'],
                composed_calls=sides['composed']['calls'] if not same['calls'] else None,
                frozen_calls=sides['frozen']['calls'] if not same['calls'] else None)


class SelectionDoubles:
    """Deterministic collaborators for the selection curve."""

    def __init__(self):
        self.calls = []

    def preparation(self):
        def prepare(spec, provenance):
            self.calls.append(dict(kind='prepare', step=spec['source_step']))
            return {'manifest_sha256': digest(f"prepare:{spec['source_step']}")}

        def audit(spec, provenance):
            self.calls.append(dict(kind='audit', step=spec['source_step']))
            return {'status': 'selection_package_audited_not_executed'}
        return SimpleNamespace(prepare=prepare, audit=audit)

    def curve(self):
        def collect(c, provenance):
            self.calls.append(dict(kind='collect', output=c['output']))
            return {'status': 'selection_curve_collected'}
        return SimpleNamespace(collect=collect)

    def operation(self, infonce, grpo):
        def run(point, ops, provenance, queue, priority, *, producer=False, authorization):
            spec = ops['selection_package']
            expected = grpo if ops['tag'].startswith('grpo') else infonce
            target = Path(ops['operations_root'])/'finalization'
            target.mkdir(parents=True, exist_ok=True)
            commit_json(target/'audit-config.json',
                        dict(tag=ops['tag'], source_step=spec['source_step'], ordered=True))
            self.calls.append(dict(kind='operation', tag=ops['tag'], step=spec['source_step'],
                                   priority=priority, producer=producer,
                                   authorization=authorization))
            return dict(status=expected, source_step=spec['source_step'],
                        runtime_provenance_verified=True)
        return run


def selection_case(config, work):
    """Frozen rewritten curve versus composed curve on identical collaborators."""
    steps = [0, 8, 318]
    template = json.loads(Path(config['sources']['selection']['path']).read_text())['point_template']
    sides = {}
    for side in ('frozen', 'composed'):
        root = Path(work)/f'selection-{side}'
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        doubles = SelectionDoubles()
        preparation, curve = doubles.preparation(), doubles.curve()
        operation = doubles.operation('complete_replica_infonce_selection_audited',
                                      'complete_replica_cached_grpo_selection_audited')
        queue = Queue()
        # The curve reads its topology from the controller config itself
        # (``c['deployment']``), so the run root is overridden rather than
        # spread, and the frozen config shape reaches both sides unchanged.
        stage = dict(config, output=str(root/'out'),
                     prepared_root=str(root/'packages'), operations_root=str(root/'jobs'),
                     remote_root=str(root/'remote'), tag_prefix='jev-differential',
                     evaluation_workers=2,
                     preprepared_stage1_endpoint=dict(package='/immutable/endpoint',
                                                      remote_root='/immutable/endpoint',
                                                      manifest_sha256='b'*64))
        if side == 'frozen':
            point = SimpleNamespace(validate=None, launch=None, audit=None, run=None)
            engine = SimpleNamespace(preparation=lambda: preparation, reconstruct=None)
            function, _ = admission_engine.optimized_selection(engine)
            collector = SimpleNamespace(collect=lambda c, p: curve.collect(c, p))
            # The frozen rewrite's own injections are replaced by the same
            # collaborators the composed path is handed; the code object stays
            # the rewrite's.
            function = FunctionType(function.__code__, {
                **function.__globals__, 'optimized_preparation': preparation,
                'optimized_point': point, 'optimized_curve': collector,
                'operation': operation})
            try:
                outcome = function(stage, template, steps, {'p': 1}, queue, grpo=False)
            except Exception as error:                    # noqa: BLE001 - recorded
                outcome = dict(error=f'{type(error).__name__}: {error}')
        else:
            point = SimpleNamespace(validate=None, launch=None, audit=None, run=None)
            try:
                outcome = composed.selection_curve(
                    stage, template, steps, {'p': 1}, queue, preparation=preparation,
                    point=point, curve=curve, operation=operation, grpo=False)
            except Exception as error:                    # noqa: BLE001 - recorded
                outcome = dict(error=f'{type(error).__name__}: {error}')
        published = {}
        output = root/'out'/'infonce'
        if output.exists():
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    published[str(path.relative_to(output))] = normalise(
                        path.read_text(), root)
        # The curve fans out over a thread pool, so call and queue order is not
        # fixed by the code under test; the multiset is, and that is what both
        # sides must agree on.
        def ordered(items):
            return sorted(normalise(items, root),
                          key=lambda item: json.dumps(item, sort_keys=True))
        sides[side] = dict(outcome=normalise(outcome, root), published=published,
                           calls=ordered(doubles.calls), queue=ordered(queue.events))
    same = {key: sides['frozen'][key] == sides['composed'][key]
            for key in ('outcome', 'published', 'calls', 'queue')}
    # Two sides that failed the same way are equal but prove nothing, so the
    # case additionally requires the curve to have been walked to the end.
    outcome = sides['composed']['outcome']
    written = outcome[0] if isinstance(outcome, (tuple, list)) else outcome
    evaluations = written.get('evaluations', []) if isinstance(written, dict) else []
    files = set(sides['composed']['published'])
    exercised = (isinstance(written, dict) and 'error' not in written
                 and [entry['step'] for entry in evaluations] == list(steps)
                 and sorted(call['kind'] for call in sides['composed']['calls'])
                 == ['audit', 'collect', 'operation', 'operation', 'operation',
                     'prepare', 'prepare']
                 and sides['composed']['queue'] == ['check', 'check', 'check']
                 and {'template.json', 'curve-config.json', 'terminal.json',
                      *(f'completed-{step:04d}.json' for step in steps)} <= files)
    return dict(steps=steps, identical=all(same.values()) and exercised,
                differences=same, exercised=exercised, entries=len(evaluations),
                files=sorted(files),
                composed=sides['composed'] if not all(same.values()) else None,
                frozen=sides['frozen'] if not all(same.values()) else None)


def admission_case(config):
    """The composed entry must return the frozen admission result unchanged.

    Delegation is read from the composed module's own source: an admission
    predicate that was copied rather than delegated would be a second
    definition, which is what this candidate must not have.
    """
    from_frozen = admission_engine.validate(config)
    from_composed = composed.validate(config)
    tree = ast.parse(Path(composed.__file__).read_text())
    delegated = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef) and node.name == 'validate'
                and isinstance(node.body[-1], ast.Return)
                and isinstance(node.body[-1].value, ast.Call)
                and isinstance(node.body[-1].value.func, ast.Name)
                and node.body[-1].value.func.id == 'admission'):
            imports = [item for item in ast.walk(node) if isinstance(item, ast.ImportFrom)
                       and item.module == 'tools.reports.context_audit_optimized_pipeline'
                       and [alias.name for alias in item.names] == ['validate']]
            delegated = bool(imports)
    return dict(identical=from_frozen == from_composed, delegated=delegated,
                stages=sorted(from_composed))


def seam_case(config):
    """Frozen ``optimized_selection`` bindings versus the composed bindings."""
    preparation = SimpleNamespace(prepare=lambda *args: None, audit=lambda *args: None)
    engine = SimpleNamespace(preparation=lambda: preparation, reconstruct=lambda c: None)
    function, operations = admission_engine.optimized_selection(engine)
    del function
    built = composed.selection_operations(preparation)
    rows = []
    for name in ('validate', 'launch', 'audit', 'run'):
        mine, theirs = getattr(built, name), getattr(operations, name)
        rows.append(dict(name=name, same_code=mine.__code__ is theirs.__code__,
                         same_names=mine.__code__.co_names == theirs.__code__.co_names,
                         injected=sorted(set(mine.__globals__) ^ set(theirs.__globals__))))
    reconstruct, collector = composed.selection_collector(
        curve_module, point=built, reconstruct_plan=engine.reconstruct)
    return dict(rows=rows, identical=all(row['same_code'] and row['same_names']
                                         for row in rows),
                reconstruct_same_code=reconstruct.__code__ is curve_module.reconstruct.__code__,
                collector_present=hasattr(collector, 'collect'))


def dispatch_flags(config):
    """Dispatch stays refused; a refusal is the only accepted outcome."""
    refused = {}
    for label, override, argument in (
            ('config_dispatch_allowed', dict(config, dispatch_allowed=True), False),
            ('explicit_dispatch_argument', dict(config, dispatch_allowed=False), True)):
        try:
            composed.run(override, {'p': 1}, dispatch=argument)
            refused[label] = None
        except ValueError as error:
            refused[label] = str(error)
    return dict(refused=refused, all_refused=all(refused.values()))


def inventory():
    """The composed module's own rewrite and injection sites.

    Injection is admissible only through ``bind``, which rebuilds a function on
    the frozen code object. A ``FunctionType`` anywhere else would mean the
    composed path synthesised behaviour.
    """
    path = Path(composed.__file__)
    sites, injections = rewriting_sites(path)
    tree = ast.parse(path.read_text())
    function = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == 'bind']
    if len(function) != 1:
        raise ValueError('exactly one bind helper required')
    body = function[0]
    calls = [node.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == 'FunctionType']
    outside = [line for line in calls if not body.lineno <= line <= body.end_lineno]
    return dict(path=str(path), sha256=file_hash(path), rewrite_sites=sites,
                injection_sites=injections, function_type_sites=sorted(calls),
                bind_body=(body.lineno, body.end_lineno),
                function_type_outside_bind=outside,
                bind_is_only_injection=not outside,
                no_text_rewriting=not sites)


def run(config, provenance):
    # The frozen controller config is carried verbatim so the admission chain
    # receives it unchanged; this audit declares itself under its own key.
    if config.get('audit_schema') != SCHEMA:
        raise ValueError('explicit composed differential schema required')
    work = Path(config['work'])
    work.mkdir(parents=True, exist_ok=False)
    commit_json(work/'config.json', config)
    commit_json(work/'provenance.json', provenance)
    failures = []

    described = inventory()
    if not described['no_text_rewriting']:
        failures.append('composed module still rewrites source: '
                        + ', '.join(described['rewrite_sites']))
    if not described['bind_is_only_injection']:
        failures.append('FunctionType outside bind: '
                        + repr(described['function_type_sites']))

    evidence, relocated = evidence_root(
        work/'evidence',
        json.loads((Path(config['preprepared_stage1_endpoint']['package'])/'config.json').read_text()),
        config['deployment']['release_evidence']['destination'])
    argv = argv_cases(config, evidence)
    for case in argv:
        if not case['identical']:
            failures.append(f"argv differs: {case['name']} producer={case['producer']}")

    prior = {name: json.loads(Path(value['path']).read_text())
             for name, value in config['sources'].items()}
    lifecycle = []
    for name, package, producer in (
            ('stage1_endpoint', config['preprepared_stage1_endpoint']['package'], True),
            ('finite_tail', prior['finite']['stages']['tail']['package'], False)):
        if not Path(package).exists():
            lifecycle.append(dict(name=name, exists=False, identical=False))
            failures.append('frozen package missing: ' + package)
            continue
        case = lifecycle_case(name, package, config, work, producer, evidence)
        lifecycle.append(case)
        if not case['identical']:
            failures.append('lifecycle differs: ' + name + ' '
                            + repr([key for key, value in case['differences'].items()
                                    if not value]))

    selection = selection_case(config, work)
    if not selection['identical']:
        differ = [key for key, value in selection['differences'].items() if not value]
        failures.append('selection curve differs: '
                        + (repr(differ) if differ
                           else 'both sides equal but the curve was not exercised'))

    chain = admission_case(config)
    if not chain['identical'] or not chain['delegated']:
        failures.append('admission chain is not the frozen chain unchanged')

    seams = seam_case(config)
    if not seams['identical'] or not seams['reconstruct_same_code']:
        failures.append('seam bindings differ from the frozen rewrite')

    flags = dispatch_flags(config)
    if not flags['all_refused']:
        failures.append('composed pipeline did not refuse dispatch')

    debt = repository_debt(config['root'])
    report = dict(
        schema=SCHEMA, status='rejected' if failures else 'accepted',
        model_calls=0, npu_executed=False, dispatch_authorised=False, failures=failures,
        environment=dict(python=sys.version.split()[0], host=platform.node(),
                         machine=platform.machine()),
        implementation_sha256=file_hash(__file__), provenance=provenance,
        scope=dict(basis='cpu-computable frozen inputs; v0.0.49 published no terminal products',
                   relocated_release_pins=relocated,
                   compare=['docker argv per real prepared package and producer mode',
                            'launch wrapper under one recording effect set',
                            'selection curve publication files, call and queue multiset',
                            'admission chain result and delegation',
                            'seam bindings and code objects'],
                   not_compared=['NPU execution', 'ssh/docker/scp transport',
                                 'collection transfer', 'downstream numerical audits']),
        candidate=described, argv=argv, lifecycle=lifecycle, selection=selection,
        admission=chain, seams=seams, dispatch=flags, debt=debt)
    commit_json(work/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    report = run(config, provenance)
    print(json.dumps(dict(status=report['status'], failures=report['failures'],
                          argv=[case['identical'] for case in report['argv']],
                          lifecycle=[case['identical'] for case in report['lifecycle']],
                          selection=report['selection']['identical'],
                          admission=report['admission']['identical'],
                          seams=report['seams']['identical'],
                          debt=report['debt']['rewrite_sites']), indent=2))
    if report['status'] != 'accepted':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
