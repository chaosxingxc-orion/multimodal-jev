from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pytest
from multimodal_jev.audit_singleflight import AuditSingleFlight


def test_concurrent_identical_work_runs_once_and_results_are_isolated():
    cache=AuditSingleFlight();entered=threading.Event();release=threading.Event();calls=[]
    def compute():calls.append(1);entered.set();assert release.wait(3);return {'proof':[1]}
    with ThreadPoolExecutor(4) as pool:
        first=pool.submit(cache.get,'a'*64,compute);assert entered.wait(3)
        rest=[pool.submit(cache.get,'a'*64,compute) for _ in range(3)]
        release.set();results=[first.result(),*[f.result() for f in rest]]
    assert calls==[1] and cache.executions==1 and cache.hits==3
    results[0][0]['proof'].append(2)
    assert cache.get('a'*64,compute)[0]=={'proof':[1]}
    assert cache.get('b'*64,lambda:{'different':True})[0]=={'different':True}


def test_failures_never_become_pass_or_implicit_retry():
    cache=AuditSingleFlight();calls=[]
    def fail():calls.append(1);raise ValueError('bad evidence')
    for _ in range(2):
        with pytest.raises(ValueError,match='bad evidence'):cache.get('a'*64,fail)
    assert calls==[1] and cache.executions==0
