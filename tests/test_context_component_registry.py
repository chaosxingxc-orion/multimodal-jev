import importlib

import pytest

from multimodal_jev import context_component_registry as components
from multimodal_jev import context_objective_registry as objectives


def test_objective_names_are_the_registered_literal_set():
    assert objectives.names() == ('candidate_infonce', 'dual_credit_grpo_floor_clipped_v1',
                                  'dual_credit_rloo_clipped_v1', 'rloo_v1')
    assert objectives.names(stage=1) == ('candidate_infonce',)
    assert objectives.names(stage=2) == ('dual_credit_grpo_floor_clipped_v1',
                                         'dual_credit_rloo_clipped_v1', 'rloo_v1')
    assert objectives.names(status='design_only') == ('rloo_v1',)


def test_resolved_objective_is_the_very_object_a_direct_import_yields():
    # Identity, not equality: a registry that rebuilt or wrapped the function
    # would drift from the trainer's own import without any test noticing.
    assert objectives.resolve('candidate_infonce') is importlib.import_module(
        'multimodal_jev.context_infonce').loss
    hybrid = importlib.import_module('multimodal_jev.context_dual_credit').hybrid_loss
    for name in ('dual_credit_grpo_floor_clipped_v1', 'dual_credit_rloo_clipped_v1'):
        assert objectives.resolve(name) is hybrid


def test_registered_algorithm_names_exist_in_the_dual_credit_module():
    # Only Stage2 names: the Stage1 objective's algorithm string is the trainer's,
    # not one the dual-credit module could ever receive.
    dual = importlib.import_module('multimodal_jev.context_dual_credit')
    registered = {entry['algorithm'] for entry in objectives.OBJECTIVES
                  if entry['stage'] == 2 and entry['algorithm'] is not None}
    assert registered == set(dual.ALGORITHMS)
    assert 'candidate_infonce' not in registered


def test_design_only_objective_has_no_callable_and_is_refused():
    found = objectives.entry('rloo_v1')
    assert found['status'] == 'design_only' and found['module'] is None
    with pytest.raises(ValueError, match='design_only'):
        objectives.resolve('rloo_v1')
    assert objectives.available('rloo_v1') is False


def test_unknown_objective_names_are_refused_before_any_work():
    for name in ('CandidateInfoNCE', 'dual_credit_grpo_floor_clipped_v2', '', None):
        with pytest.raises(ValueError, match='unregistered objective'):
            objectives.resolve(name)
        with pytest.raises(ValueError, match='unregistered objective'):
            objectives.entry(name)


def test_entry_is_a_copy_so_a_caller_cannot_edit_the_registry():
    first = objectives.entry('candidate_infonce')
    first['module'] = 'multimodal_jev.context_dual_credit'
    assert objectives.entry('candidate_infonce')['module'] == 'multimodal_jev.context_infonce'
    assert objectives.resolve('candidate_infonce').__module__ == 'multimodal_jev.context_infonce'


def test_dependency_marker_reports_host_availability_without_importing():
    # The dual-credit surrogate calls the pinned verl loss; on a CPU-only host the
    # marker says so instead of raising during validation.
    found = objectives.entry('dual_credit_grpo_floor_clipped_v1')
    assert found['requires'] == ('omegaconf', 'verl')
    assert type(objectives.available('dual_credit_grpo_floor_clipped_v1')) is bool
    assert objectives.available('candidate_infonce') is True


def test_catalog_reports_a_digest_for_every_written_module():
    rows = {name: (path, digest, status, ok)
            for name, _, _, path, digest, status, ok in objectives.catalog()}
    for name in objectives.names(status='admitted'):
        path, digest, status, ok = rows[name]
        # ok is host availability, not admission: a Stage2 objective on a host
        # without verl is admitted and still reported as not runnable here.
        assert status == 'admitted' and ok is objectives.available(name)
        assert path.endswith('.py') and len(digest) == 64
    assert rows['candidate_infonce'][3] is True
    assert rows['rloo_v1'] == (None, None, 'design_only', False)


def test_component_kinds_are_the_registered_literal_set():
    assert components.KINDS == ('release', 'objective', 'evaluator', 'topology', 'controller')
    assert components.names('release') == ('slurp_portable_release_v2', 'release_handle_v1')
    assert components.names('evaluator') == ('context_event_metrics_v1',
                                             'context_evaluation_contract_v1')
    assert components.names('topology') == ('context_topology_batch_plan_v1',)
    with pytest.raises(ValueError):
        components.names('trainer')


def test_objective_kind_defers_to_the_objective_registry():
    assert components.names('objective') == objectives.names()
    found = components.entry('objective', 'candidate_infonce')
    assert found['kind'] == 'objective' and found['module'] == 'multimodal_jev.context_infonce'
    assert components.resolve('objective', 'candidate_infonce') is objectives.resolve(
        'candidate_infonce')


def test_every_admitted_component_resolves_to_a_real_object():
    for kind in ('release', 'objective', 'evaluator', 'topology', 'controller'):
        for name in components.names(kind):
            if components.entry(kind, name)['status'] != 'admitted':
                continue
            assert components.resolve(kind, name) is not None, f'{kind}/{name}'


def test_composed_controller_is_admitted_but_refuses_dispatch():
    found = components.entry('controller', 'context_composed_pipeline_v1')
    assert found['status'] == 'admitted'
    assert found['module'] == 'tools.reports.context_composed_pipeline'
    run = components.resolve('controller', 'context_composed_pipeline_v1')
    assert run is importlib.import_module('tools.reports.context_composed_pipeline').run
    # Admission is resolution, not authorisation: the candidate stays
    # construction-only until a versioned successor is published.
    with pytest.raises(ValueError, match='construction-only'):
        run({'dispatch_allowed': True}, {'p': 1})
    assert components.catalog().count(('controller', 'context_composed_pipeline_v1',
                                       None, None)) == 0


def test_unknown_components_are_refused_without_importing_anything():
    for kind, name in (('release', 'slurp_release_v3'), ('objective', 'ppo_v1'),
                       ('evaluator', 'context_event_metrics_v2'), ('oracle', 'anything')):
        with pytest.raises(ValueError):
            components.resolve(kind, name)


def test_validate_resolves_the_whole_config_before_any_work_starts():
    resolved = components.validate(dict(
        release='slurp_portable_release_v2', objective='candidate_infonce',
        evaluator='context_event_metrics_v1', topology='context_topology_batch_plan_v1'))
    assert set(resolved) == {'release', 'objective', 'evaluator', 'topology'}
    assert all(callable(value) or isinstance(value, type) for value in resolved.values())
    with pytest.raises(ValueError, match='unregistered component kinds'):
        components.validate(dict(release='slurp_portable_release_v2', optimizer='adamw'))
    with pytest.raises(ValueError):
        components.validate(dict(release='slurp_release_v3'))
    with pytest.raises(ValueError):
        components.validate({})


def test_topology_seam_still_refuses_unmeasured_microbatches():
    derive = components.resolve('topology', 'context_topology_batch_plan_v1')
    base = dict(training_devices=4, tensor_parallel=1, pipeline_parallel=1,
                per_dp_replica_batch=64, microbatch=16, inference_instances=2,
                inference_batch_queries=32, source_queries=40513, epochs=1,
                efficiency_steps=100)
    assert derive(base)['global_batch'] == 256
    with pytest.raises(ValueError):
        derive(dict(base, microbatch=4))