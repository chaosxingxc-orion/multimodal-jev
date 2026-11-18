"""Process-local proofs bound to freshly hashed observed read dependencies.

Only explicitly registered deterministic auditors may use this session. Reads
from subprocesses/network are outside this contract. A registered auditor must
read/hash its subprocess inputs in this process (the source batch auditor does).
No disk PASS cache or stat-only integrity shortcut. Directory membership and
code bytes participate in reuse. Never use around a launcher or training loop.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
import threading
import time

_ACTIVE = None
_LOCAL = threading.local()
_INSTALLED = False


def _hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _hook(event,args):
    session=_ACTIVE
    if session is None or getattr(_LOCAL,'busy',False):return
    if event not in ('open','os.scandir','os.listdir'):return
    path=args[0]
    if not isinstance(path,(str,bytes,os.PathLike)):return
    path=os.path.abspath(os.fsdecode(path))
    # Device/proc reads describe runtime, never immutable evidence.
    if path.startswith(('/proc/','/dev/','/sys/')):return
    if event=='open':
        mode,flags=args[1:3]
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
            if path in session.files:raise ValueError('auditor attempted to write an input dependency: '+path)
            session.writes.add(path);return
    _LOCAL.busy=True
    try:
        p=Path(path)
        if p.is_symlink():raise ValueError('proof dependency symlink: '+path)
        if event=='open' and p.is_file():
            with session.lock:
                if path not in session.files:session.files[path]=_hash(path)
        elif event!='open' and p.is_dir():
            with session.lock:
                if path not in session.directories:session.directories[path]=tuple(sorted(os.listdir(path)))
    finally:_LOCAL.busy=False


class ContentProofSession:
    def __init__(self,code_paths,*,io_workers=4):
        global _INSTALLED
        if not _INSTALLED:sys.addaudithook(_hook);_INSTALLED=True
        self.code_paths=tuple(sorted(str(Path(p).absolute()) for p in code_paths))
        self.io_workers=io_workers;self.entries={};self.events=[]
        self.lock=threading.RLock();self.files={};self.directories={};self.writes=set()

    def identity(self,files,dirs):
        paths=sorted(set(files)|set(self.code_paths))
        def one(p):
            if Path(p).is_symlink() or not Path(p).is_file():raise ValueError('proof dependency changed: '+p)
            return p,_hash(p)
        with ThreadPoolExecutor(max_workers=self.io_workers) as pool:actual=dict(pool.map(one,paths))
        actual_dirs={p:tuple(sorted(os.listdir(p))) for p in dirs}
        if any(Path(p).is_symlink() for p in dirs):raise ValueError('proof directory replaced')
        return actual,actual_dirs

    def get(self,key,compute):
        global _ACTIVE
        if _ACTIVE is not None:raise ValueError('nested proof capture forbidden')
        started=time.perf_counter();entry=self.entries.get(key)
        if entry is not None:
            files,dirs,result=entry
            now=self.identity(files,dirs)
            if now!=(files,dirs):raise ValueError('frozen proof dependencies changed; new session required')
            self.events.append(dict(node=key,cache_hit=True,seconds=time.perf_counter()-started,files=len(files)))
            return deepcopy(result)
        self.files={};self.directories={};self.writes=set()
        initial_code=self.identity({}, {})[0]
        _ACTIVE=self
        try:result=compute()
        finally:_ACTIVE=None
        # Temporary reconstruction files may be written/read/removed by an
        # auditor. They are derived outputs, not reusable input dependencies.
        files={p:h for p,h in self.files.items() if p not in self.writes}
        dirs={p:v for p,v in self.directories.items() if Path(p).is_dir()
              and not any(Path(w).is_relative_to(p) for w in self.writes)}
        files.update(initial_code)
        now=self.identity(files,dirs)
        if now!=(files,dirs):
            changed=[p for p in set(files)|set(now[0]) if files.get(p)!=now[0].get(p)]
            changed += [p for p in set(dirs)|set(now[1]) if dirs.get(p)!=now[1].get(p)]
            raise ValueError('proof dependencies changed during computation: '+repr(changed[:12]))
        self.entries[key]=(files,dirs,deepcopy(result))
        self.events.append(dict(node=key,cache_hit=False,seconds=time.perf_counter()-started,files=len(files)))
        return deepcopy(result)
