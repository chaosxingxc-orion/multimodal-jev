import pytest
from tools.reports.context_audit_release import fast_launch


def test_unadmitted_old_launcher_refuses_dispatch_before_any_validation():
    calls=[]
    with pytest.raises(RuntimeError,match='retired source-rewriting launcher'):
        fast_launch(lambda *a:calls.append(a),None,None,None)
    assert calls==[]
