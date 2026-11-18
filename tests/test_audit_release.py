from types import SimpleNamespace
from tools.reports.context_audit_release import graph,acceptance_graph
import pytest


def test_unaccepted_full_graph_cannot_dispatch():
    with pytest.raises(RuntimeError,match='admission remains incomplete'):
        acceptance_graph(SimpleNamespace())


def test_new_graph_reuses_both_proofs_without_mutating_old_graph():
    from tools.reports import prepare_objective_training_candidate as original
    from tools.reports import prepare_context_reference_job as reference
    before=original.expected;old_ref=reference.replay_reference_outputs
    engine=SimpleNamespace(reconstruct=lambda *a:None,replay=lambda *a:None)
    views=graph(engine);p=views['tools.reports.prepare_objective_training_candidate']
    assert p.audit.__globals__['expected'] is p.expected
    assert p.prepare.__globals__['expected'] is p.expected
    assert views['tools.reports.prepare_context_reference_job'].audit_results.__globals__['replay_reference_outputs'] is engine.replay
    assert original.expected is before and reference.replay_reference_outputs is old_ref
