from concurrent.futures import ThreadPoolExecutor
import os
import pytest
from tools.reports.context_isolated_source_audit import IsolatedSourceAuditEngine


def test_owned_process_persists_and_close_reaps():
    with IsolatedSourceAuditEngine(cpu_workers=1, io_workers=1) as engine:
        with ThreadPoolExecutor(max_workers=2) as callers:
            results=list(callers.map(lambda _:engine.statistics(), range(2)))
        assert results[0] == results[1]
        pid=results[0]['pid']
        assert pid != os.getpid()
    with pytest.raises(ProcessLookupError):os.kill(pid, 0)
    with pytest.raises(RuntimeError, match='closed'):engine.statistics()


def test_worker_failure_propagates_without_retry():
    with IsolatedSourceAuditEngine(cpu_workers=1, io_workers=1) as engine:
        with pytest.raises(ValueError, match='unregistered'):engine.call('unknown')
        with pytest.raises(RuntimeError, match='automatic retry'):engine.statistics()


def test_actual_audit_error_propagates():
    with IsolatedSourceAuditEngine(cpu_workers=1, io_workers=1) as engine:
        with pytest.raises(KeyError):engine.reconstruct({})
        assert engine.failed


@pytest.mark.parametrize('workers',[0,True,33])
def test_worker_budget_rejected(workers):
    with pytest.raises(ValueError):IsolatedSourceAuditEngine(cpu_workers=workers)
