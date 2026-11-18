import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from multimodal_jev.context_resident_rollout import FixedContextRollout
from tools.reports import context_efficiency_owner_candidate as candidate

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'tools/jev/context_switch_rollout.py'


def test_source_sha_pin_matches_frozen_file():
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    assert digest == candidate.SOURCE_SHA == (
        'beb755c48b4307a0754204a3aaae0fff5aba30187f1d70957894b208cb1c0813')


def test_transformed_source_applies_union_and_compiles():
    source = candidate.transformed_source(ROOT)
    assert compile(source, '<efficiency>', 'exec')
    assert "max_num_seqs=config['max_batch_queries']" in source
    assert "max_num_batched_tokens=config['max_batch_tokens']" in source
    assert ('worker_extension_cls='
            "'tools.reports.context_batched_worker_extension_candidate.BatchedContextWorkerCandidate'"
            ) in source
    assert 'gpu_memory_utilization=.6' in source
    assert 'gpu_memory_utilization=.4' not in source
    # The lifecycle after construction is sourced unchanged from the frozen file.
    original = FROZEN.read_text()
    assert source[source.index('    def switch('):] == original[original.index('    def switch('):]


def test_gpu_memory_utilization_replaced_exactly_once():
    original = FROZEN.read_text()
    assert original.count('gpu_memory_utilization=.4') == 1
    source = candidate.transformed_source(ROOT)
    assert source.count('gpu_memory_utilization=') == 1
    assert source.count('gpu_memory_utilization=.6') == 1


def test_source_hash_is_reverified_and_tampered_hash_rejected(tmp_path):
    target = tmp_path / 'tools/jev/context_switch_rollout.py'
    target.parent.mkdir(parents=True)
    target.write_text(FROZEN.read_text() + '\n')
    with pytest.raises(ValueError, match='source changed'):
        candidate.transformed_source(tmp_path)


def test_duplicate_memory_utilization_literal_rejected(tmp_path, monkeypatch):
    doubled = FROZEN.read_text().replace(
        'gpu_memory_utilization=.4,',
        'gpu_memory_utilization=.4,gpu_memory_utilization=.4,')
    target = tmp_path / 'tools/jev/context_switch_rollout.py'
    target.parent.mkdir(parents=True)
    target.write_text(doubled)
    monkeypatch.setattr(candidate, 'SOURCE_SHA', hashlib.sha256(doubled.encode()).hexdigest())
    with pytest.raises(ValueError, match='unexpected frozen owner construction'):
        candidate.transformed_source(tmp_path)


def test_generate_returns_the_same_valid_source():
    assert candidate.generate(ROOT) == candidate.transformed_source(ROOT)


def test_owner_class_derives_the_efficiency_owner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools/jev'))
    owner = candidate.owner_class(ROOT)
    assert owner.__name__ == 'EfficiencyContextOwnerCandidate'
    assert issubclass(owner, FixedContextRollout)


def test_efficiency_config_token_budget():
    config = json.loads((ROOT / 'configs/experiments/context-efficiency-v1.json').read_text())
    service = config['efficiency_service']
    assert service['schema'] == 'jev-efficiency-service-v1'
    assert service['service_max_batch_tokens'] == 65536
    assert service['service_max_batch_queries'] == 48
    assert service['gpu_memory_utilization'] == 0.6
    assert service['gpu_memory_utilization_from'] == 0.4
    assert service['source_sha256'] == candidate.SOURCE_SHA
    assert service['candidate_runtime'] == candidate.RUNTIME
    assert service['max_steps'] == 100
    assert service['training_dispatch_allowed'] is False
    # The pre-registered design fields are preserved, not overwritten.
    assert config['changes']['engine_token_budget']['to'] == 65536
    assert config['changes']['engine_memory_utilization']['to'] == 0.6
    assert config['efficiency_budget']['max_steps'] == 100
    assert config['training_dispatch_allowed'] is False
