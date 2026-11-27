"""Training objectives by name; a config selects an algorithm, never a module path.

Resolution imports the module and returns the very object a direct import would
yield, so a registered name cannot drift to a different implementation without a
test failing. The table is a literal tuple, not a directory scan: a typo is an
error at validation time, not a silently missing objective at update 300.

Two entries share the dual-credit loss because the algorithm name is carried in
each record's binding (``dual_credit.ALGORITHMS``), not passed to the loss. What
the registry adds is the check that a config names an algorithm the trainer can
actually run, and an honest dependency marker: the dual-credit surrogate calls
the pinned verl loss, which is absent on hosts that do not run the Stage2 engine.
"""
import copy
import importlib
import importlib.util
from pathlib import Path

from .baseline import file_hash

SCHEMA = 'jev-context-objective-registry-v1'
STATUSES = ('admitted', 'design_only')

OBJECTIVES = (
    dict(name='candidate_infonce', module='multimodal_jev.context_infonce',
         attribute='loss', stage=1, algorithm='candidate_infonce', requires=(),
         status='admitted',
         summary='Supervised candidate InfoNCE; the softmax is the training objective '
                 'only, never the inference readout.'),
    dict(name='card_bidirectional_infonce_v1', module='multimodal_jev.card_contrast',
         attribute='components', stage=1, algorithm='card_bidirectional_infonce_v1', requires=('torch',),
         status='admitted', summary='Membership-logit InfoNCE over wrong cards and wrong recordings.'),
    dict(name='card_and_speech_infonce_v1', module='multimodal_jev.card_contrast',
         attribute='components', stage=1, algorithm='card_and_speech_infonce_v1', requires=('torch',),
         status='admitted', summary='Executable Stage1 objective: full-context membership and pre-card speech contrast.'),
    dict(name='dual_credit_grpo_floor_clipped_v1', module='multimodal_jev.context_dual_credit',
         attribute='hybrid_loss', stage=2, algorithm='dual_credit_grpo_floor_clipped_v1',
         requires=('omegaconf', 'verl'), status='admitted',
         summary='Dual-credit RL with floor-clipped GRPO advantages, plus the fixed '
                 'ranking/probability supervision and reference KL terms.'),
    dict(name='dual_credit_rloo_clipped_v1', module='multimodal_jev.context_dual_credit',
         attribute='hybrid_loss', stage=2, algorithm='dual_credit_rloo_clipped_v1',
         requires=('omegaconf', 'verl'), status='admitted',
         summary='Same-budget RLOO-clipped control for the dual-credit comparison.'),
    dict(name='rloo_v1', module=None, attribute=None, stage=2, algorithm=None, requires=(),
         status='design_only',
         summary='Reserved name for the plain-RLOO design; no implementation is registered.'),
)

FIELDS = ('name', 'module', 'attribute', 'stage', 'algorithm', 'requires', 'status', 'summary')


def _indexed():
    if len({entry['name'] for entry in OBJECTIVES}) != len(OBJECTIVES):
        raise ValueError('duplicate objective name in the registry')
    for entry in OBJECTIVES:
        if set(entry) != set(FIELDS) or entry['status'] not in STATUSES:
            raise ValueError('every objective entry needs exactly the declared fields')
        if entry['status'] == 'design_only' and (entry['module'] is not None
                                                 or entry['attribute'] is not None):
            raise ValueError('design-only objectives carry no callable')
    return {entry['name']: entry for entry in OBJECTIVES}


_BY_NAME = _indexed()


def names(*, stage=None, status=None):
    """Registered objective names, optionally filtered; deterministic order."""
    if stage is not None and type(stage) is not int:
        raise ValueError('integer stage required')
    if status is not None and status not in STATUSES:
        raise ValueError('known objective status required')
    return tuple(entry['name'] for entry in OBJECTIVES
                 if (stage is None or entry['stage'] == stage)
                 and (status is None or entry['status'] == status))


def entry(name):
    """A copy of one literal entry; never the mutable original."""
    if name not in _BY_NAME:
        raise ValueError('unregistered objective: ' + str(name))
    return copy.deepcopy(_BY_NAME[name])


def module_file(module):
    """Where a dotted module lives, without importing it."""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ValueError('registered module is not importable: ' + module)
    return Path(spec.origin)


def available(name):
    """Whether this objective could actually run on this host.

    A design-only entry is not available however few dependencies it declares: it
    has no callable, so "every requirement imports" would be vacuously true and
    the marker would report a runnable objective that does not exist.
    """
    found = entry(name)
    if found['module'] is None:
        return False
    for requirement in found['requires']:
        if importlib.util.find_spec(requirement) is None:
            return False
    return True


def resolve(name, *, statuses=('admitted',)):
    """Import and return the objective object itself, not a wrapper or a copy."""
    found = entry(name)
    if found['status'] not in statuses:
        raise ValueError(f"{name} is {found['status']}, not resolvable as {list(statuses)}")
    if found['module'] is None:
        raise ValueError('design-only objective has no callable: ' + name)
    value = getattr(importlib.import_module(found['module']), found['attribute'], None)
    if not callable(value):
        raise ValueError(f"{found['module']}.{found['attribute']} is missing or not callable")
    return value


def catalog():
    """(name, module, attribute, module_file, sha256, status, available) per objective.

    A design-only entry has no file or digest; that absence is reported rather
    than filled with a placeholder a reader might mistake for provenance.
    """
    rows = []
    for found in OBJECTIVES:
        if found['module'] is None:
            rows.append((found['name'], None, None, None, None, found['status'], False))
            continue
        path = module_file(found['module'])
        rows.append((found['name'], found['module'], found['attribute'], str(path),
                     file_hash(path), found['status'], available(found['name'])))
    return rows
