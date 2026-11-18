"""Shared update-state machine executes; model/credit checks use existing fixtures."""
from pathlib import Path

import pytest

from test_context_macro_formal_audit import fixture
from tools.reports import audit_context_macro_training as shared
from tools.reports import audit_context_macro_formal_prefix as module


def interrupted(tmp_path, monkeypatch, fault=None):
    package, collection, checks = fixture(tmp_path, monkeypatch, fault)
    original = shared.read
    def read(path):
        path = Path(path)
        if path.name == 'result.json' or path.name == 'terminal.json':
            raise AssertionError('interrupted attempts have no successful completion receipt')
        return original(path)
    monkeypatch.setattr(shared, 'read', read)
    return package, collection, checks


def test_prefix_checks_every_consumed_update_without_requiring_successful_parent(tmp_path,monkeypatch):
    package,collection,checks=interrupted(tmp_path,monkeypatch)
    result=module.audit_prefix(package,collection,64)
    assert result['status']=='formal_dp2_prefix_outputs_audited'
    assert [u['update'] for u in result['updates']]==list(range(64))
    assert checks==[64] and len(result['macros'])==64
    assert result['training_dispatch_allowed'] is False
    assert result['fresh_process_replay_verified'] is False
    assert result['ignored_suffix'] is True


@pytest.mark.parametrize('fault',['checkpoint_policy','checkpoint_flag','extra_checkpoint',
    'tail_denominator','stale_replica','wrong_export','missing_policy'])
def test_faults_in_consumed_prefix_are_rejected(tmp_path,monkeypatch,fault):
    package,collection,_=interrupted(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):module.audit_prefix(package,collection,320)


def test_damaged_unconsumed_suffix_does_not_invalidate_earlier_committed_boundary(tmp_path,monkeypatch):
    package,collection,checks=interrupted(tmp_path,monkeypatch,'wrong_export')
    (collection/'run/policy-0317-committed.json').unlink()
    result=module.audit_prefix(package,collection,256)
    assert len(result['updates'])==256 and checks==[64,128,192,256]


@pytest.mark.parametrize('cursor',[0,1,317,634,64.0,True])
def test_only_sparse_nonterminal_boundaries(cursor):
    with pytest.raises(ValueError):module.audit_prefix('/missing','/missing',cursor)


def test_missing_consumed_journal_rejected(tmp_path,monkeypatch):
    package,collection,_=interrupted(tmp_path,monkeypatch)
    (collection/'journal/formal/batch-00000010').rmdir()
    with pytest.raises(ValueError,match='prefix macro journal'):
        module.audit_prefix(package,collection,64)


def test_dependency_changes_rejected_before_derivation(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'AUDITOR_SHA256','0'*64)
    with pytest.raises(ValueError,match='shared prefix auditor'):
        module.audit_prefix(tmp_path,tmp_path,64)
