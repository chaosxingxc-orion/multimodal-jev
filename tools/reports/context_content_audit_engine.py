"""Complete source and reference proof reuse in one owned CPU worker."""
import json
from pathlib import Path
from types import FunctionType
from multimodal_jev.audit_content_session import ContentProofSession
from tools.reports.context_isolated_source_audit import IsolatedSourceAuditEngine
from tools.reports.context_shared_source_audit import SourceAuditEngine
from tools.reports.context_shared_reference_audit import ReferenceAuditEngine
from tools.reports.context_composed_reference_audit import replay as replay_reference
from multimodal_jev.context_execution import widths
from tools.reports import context_fast_reference_inputs

_SESSION=None
_SOURCE=None
_REFERENCE=None


def initialize(cpu_workers,io_workers):
    global _SESSION,_SOURCE,_REFERENCE
    import torch
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[2]
    paths=[p for folder in ('src/multimodal_jev','tools/reports','tools/jev') for p in (root/folder).glob('*.py')]
    _SESSION=ContentProofSession(paths,io_workers=io_workers)
    _SOURCE=SourceAuditEngine(cpu_workers=cpu_workers,io_workers=io_workers)
    _REFERENCE=ReferenceAuditEngine(io_workers=io_workers)
    # The pure binder still uses all original receipt/coverage checks.
    from tools.reports import context_reference_input_evidence as original
    def load_inputs(c,**kw):
        fn=FunctionType(original.load.__code__,dict(original.load.__globals__,bind=context_fast_reference_inputs.bind))
        return fn(c,**kw)
    _REFERENCE.load_inputs=load_inputs


def execute(action,args):
    if action=='statistics':return _SESSION.events
    if action=='verify':
        for files,dirs,_ in _SESSION.entries.values():
            if _SESSION.identity(files,dirs)!=(files,dirs):raise ValueError('worker proof dependency changed')
        return True
    if action=='source':
        c=args[0]
        key=json.dumps(dict(action=action,source_plan=c['source_plan'],source_plan_sha256=c['source_plan_sha256'],
            source_finalization=c['source_finalization']),sort_keys=True)
        return _SESSION.get(key,lambda:_SOURCE.reconstruct(c))
    if action=='reference':
        c,p=args
        # Provenance describes caller, never mathematics; avoid recomputation
        # when the same evidence is checked by a different lifecycle layer.
        key=json.dumps(dict(action=action,config=c),sort_keys=True)
        result=_SESSION.get(key,lambda:replay_reference(c,{'source':'content_bound_reference'},
            load_inputs=_REFERENCE.load_inputs,audit_journal=_REFERENCE.audit))
        result['provenance']=p;return result
    raise ValueError('unknown proof action')


class ContentAuditEngine(IsolatedSourceAuditEngine):
    def __init__(self,*,cpu_workers=4,io_workers=4):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing,threading
        widths(dict(cpu_workers=cpu_workers,io_workers=io_workers))
        self.pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(cpu_workers,io_workers))
        self.lock=threading.Lock();self.failed=False;self.closed=False
    def call(self,action,*args):
        with self.lock:
            if self.failed or self.closed:raise RuntimeError('proof engine unavailable')
            try:return self.pool.submit(execute,action,args).result()
            except BaseException:self.failed=True;raise
    def reconstruct(self,c):return self.call('source',c)
    def replay(self,c,p):return self.call('reference',c,p)
    def statistics(self):return self.call('statistics')
    def verify(self):return self.call('verify')
