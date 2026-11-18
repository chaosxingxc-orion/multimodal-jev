"""Bounded in-memory deduplication of content-addressed deterministic audits.

The caller MUST compute its key from freshly read full inputs and auditor code,
never mtime/stat or a prior PASS. Failed computations are not retried implicitly.
Results are copied to prevent consumers mutating shared evidence.
"""
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
import threading


class AuditSingleFlight:
    def __init__(self, *, max_entries=4, max_running=1):
        if any(type(n)is not int or n<1 for n in (max_entries,max_running)):
            raise ValueError('positive bounded cache capacity/concurrency required')
        self.max_entries=max_entries
        self.lock=threading.Lock();self.slots=threading.Semaphore(max_running)
        self.entries=OrderedDict();self.executions=0;self.hits=0

    def get(self, key, compute):
        if type(key)is not str or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('SHA256 of complete inputs and auditor identity required')
        with self.lock:
            future=self.entries.get(key)
            owner=future is None
            if owner:
                # Never evict an in-flight computation; temporary overflow is
                # bounded by the caller's configured thread count.
                future=Future();self.entries[key]=future
            else:self.hits+=1
        if owner:
            try:
                with self.slots:
                    result=compute()
                with self.lock:self.executions+=1
                future.set_result(deepcopy(result))
            except BaseException as exc:
                future.set_exception(exc)
            with self.lock:
                while len(self.entries)>self.max_entries:
                    victim=next((k for k,v in self.entries.items() if k!=key and v.done()),None)
                    if victim is None:break
                    self.entries.pop(victim)
        return deepcopy(future.result()),not owner
