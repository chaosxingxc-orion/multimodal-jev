import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from audit_intent_queue_budget import seconds,estimate


def test_two_backends_not_double_counted_in_runtime_rate():
    r=estimate([dict(requests=178,training_seconds=60,suite_seconds=356)],22758,7)
    assert r['target_requests_per_backend']==159306 and r['target_forward_calls']==318612
    assert r['extrapolated_matrix_hours']['median']==pytest.approx(2*159306/3600)


def test_only_actual_successful_elapsed_time():
    s=dict(Status='exited',ExitCode=0,StartedAt='2026-11-12T10:00:00.000000000Z',FinishedAt='2026-11-12T10:01:30.500000000Z')
    assert seconds(s)==90.5
    for bad in [dict(s,Status='running'),dict(s,ExitCode=1),dict(s,FinishedAt=s['StartedAt'])]:
        with pytest.raises(ValueError):seconds(bad)
