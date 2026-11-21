"""The seams a config may name: release, objective, evaluator, topology, controller.

Each kind has exactly one role, and every name resolves to a real object in a real
file. A config that says ``"evaluator": "context_event_metrics_v1"`` fails at
validation if that name is not registered, instead of failing 40 minutes later
inside a rollout. Resolution never scans a directory and never imports a path
built from config text.

The objective kind is not duplicated here: it lives in
``context_objective_registry`` with its own stage and dependency metadata, and
this module defers to it so the two can never disagree.
"""
import copy
import importlib
import importlib.util
from pathlib import Path

from . import context_objective_registry
from .baseline import file_hash

SCHEMA = 'jev-context-component-registry-v1'
KINDS = ('release', 'objective', 'evaluator', 'topology', 'controller')
STATUSES = ('admitted', 'design_only')

# release: authenticated dataset publication reader.
# evaluator: the one metric accumulator plus the contract that projects it.
# topology: batch-size derivation, the existing seam every controller already uses.
# controller: the orchestration entry point a pipeline may delegate to.
COMPONENTS = (
    dict(kind='release', name='stage2_difficulty_release_v2',
         module='multimodal_jev.stage2_dataset_release', attribute='TrainingDataset',
         status='admitted', requires=(),
         summary='CPU validated TRAIN difficulty dataset, deterministic audit and batch consumer; '
                 'resolution does not authorize NPU training.'),
    dict(kind='release', name='slurp_portable_release_v2',
         module='multimodal_jev.slurp_portable_release', attribute='load',
         status='admitted', requires=(),
         summary='Portable official-clean SLURP release; validates the manifest and every '
                 'pinned dependency, and refuses role/purpose pairs it cannot honour.'),
    dict(kind='release', name='release_handle_v1',
         module='multimodal_jev.context_release_handle', attribute='ReleaseHandle',
         status='admitted', requires=(),
         summary='Content-keyed cache in front of the release reader; authenticates once per '
                 'complete content identity and reuses byte-identical evidence.'),
    dict(kind='evaluator', name='context_event_metrics_v1',
         module='multimodal_jev.context_event_metrics', attribute='ContextEventMetrics',
         status='admitted', requires=(),
         summary='Independent-sigmoid set metrics at a fixed threshold; undefined ratios stay '
                 'undefined and missing labels are excluded, never scored as negatives.'),
    dict(kind='evaluator', name='context_evaluation_contract_v1',
         module='multimodal_jev.context_evaluation_contract', attribute='from_result',
         status='admitted', requires=(),
         summary='The single metric vocabulary: field tuples, row validation, selection order '
                 'and paired differences.'),
    dict(kind='topology', name='context_topology_batch_plan_v1',
         module='multimodal_jev.context_topology_batch_plan', attribute='derive',
         status='admitted', requires=(),
         summary='Derives DP replicas, global batch, accumulation and update counts from an '
                 'explicit topology; refuses unmeasured microbatch values.'),
    dict(kind='controller', name='context_async_pipeline_v1',
         module='tools.reports.context_async_pipeline', attribute='run',
         status='admitted', requires=(),
         summary='The serial prepare/launch/audit/collect chain currently driving the queue.'),
    dict(kind='controller', name='context_composed_pipeline_v1',
         module='tools.reports.context_composed_pipeline', attribute='run',
         status='admitted', requires=(),
         summary='Composition-by-injection replacement for the source-rewriting audit '
                 'controller; reproduced the frozen outputs field by field, still '
                 'construction-only and refusing dispatch until a versioned successor '
                 'is published.'),
)

FIELDS = ('kind', 'name', 'module', 'attribute', 'status', 'requires', 'summary')


def _indexed():
    for entry in COMPONENTS:
        if set(entry) != set(FIELDS):
            raise ValueError('every component entry needs exactly the declared fields')
        if entry['kind'] not in KINDS or entry['status'] not in STATUSES:
            raise ValueError('registered kind and status required')
        if entry['status'] == 'design_only' and entry['module'] is not None:
            raise ValueError('design-only components are not importable yet')
    keys = [(entry['kind'], entry['name']) for entry in COMPONENTS]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate (kind, name) in the component registry')
    return {(entry['kind'], entry['name']): entry for entry in COMPONENTS}


_BY_KEY = _indexed()


def _objective_entries():
    """Project the objective registry into the same row shape, one source of truth."""
    return [dict(kind='objective', **context_objective_registry.entry(name))
            for name in context_objective_registry.names()]


def names(kind):
    """Every registered name of one kind, objectives included, in literal order."""
    if kind not in KINDS:
        raise ValueError('registered component kind required: ' + str(kind))
    if kind == 'objective':
        return context_objective_registry.names()
    return tuple(entry['name'] for entry in COMPONENTS if entry['kind'] == kind)


def entry(kind, name):
    """A copy of one literal entry; never the mutable original."""
    if kind == 'objective':
        return dict(kind='objective', **context_objective_registry.entry(name))
    if (kind, name) not in _BY_KEY:
        raise ValueError(f'unregistered component: {kind}/{name}')
    return copy.deepcopy(_BY_KEY[(kind, name)])


def resolve(kind, name, *, statuses=('admitted',)):
    """Import and return the component object itself, not a wrapper or a copy.

    ``statuses`` is explicit because a design-only entry has no callable: asking
    for one is an error, not an empty result.
    """
    found = entry(kind, name)
    if found['status'] not in statuses:
        raise ValueError(f'{kind}/{name} is {found["status"]}, not resolvable as {list(statuses)}')
    if found['module'] is None:
        raise ValueError(f'{kind}/{name} has no implementation yet')
    value = getattr(importlib.import_module(found['module']), found['attribute'], None)
    if value is None:
        raise ValueError(f'{found["module"]}.{found["attribute"]} is missing')
    return value


def catalog():
    """(kind, name, module_file, sha256) per component; None where not yet written."""
    rows = []
    for found in sorted(_objective_entries(), key=lambda row: (row['kind'], row['name'])):
        rows.append(_row(found))
    for found in COMPONENTS:
        rows.append(_row(found))
    return rows


def _row(found):
    if found['module'] is None:
        return (found['kind'], found['name'], None, None)
    spec = importlib.util.find_spec(found['module'])
    if spec is None or spec.origin is None:
        return (found['kind'], found['name'], None, None)
    path = Path(spec.origin)
    return (found['kind'], found['name'], str(path), file_hash(path))


def validate(config):
    """Resolve every component a config names, before any work starts."""
    if type(config) is not dict or not config:
        raise ValueError('object of kind -> name required')
    unknown = set(config) - set(KINDS)
    if unknown:
        raise ValueError('unregistered component kinds: ' + ', '.join(sorted(unknown)))
    resolved = {}
    for kind, name in config.items():
        resolved[kind] = resolve(kind, name)
    return resolved