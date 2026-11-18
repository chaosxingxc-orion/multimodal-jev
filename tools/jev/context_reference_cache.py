"""Immutable fixed-reference probability cache for Stage2, separate from inputs.

Only authenticated fixed reference records can be stored. No labels, actions or
changing behavior-policy records belong here. Caller owns engine provenance.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from multimodal_jev.context_rollout_batch import bind_rollout
from multimodal_jev.local_checkpoint import sync_directory
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.span_request_identity import tensor_identity


class ReferenceCache:
    def __init__(self,root,*,policy_id,base_id,contract):
        for value in (policy_id,base_id):
            if (not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('authenticated fixed policy/base digest required')
        if contract.get('base_id')!=base_id:raise ValueError('reference base differs')
        self.root=Path(root);check_paths({'reference_cache':self.root})
        self.root.mkdir(parents=True,exist_ok=True)
        self.identity=dict(schema='context-fixed-reference-cache-v1',policy_id=policy_id,base_id=base_id,contract=contract)
        self.policy_id=policy_id
        self.namespace=self.root/self.digest(self.identity)
        self.namespace.mkdir(exist_ok=True)

    @staticmethod
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def binding(self,inputs,positions):
        return dict(policy_id=self.policy_id,positions=positions[0].tolist(),
                    inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})

    def read(self,inputs,positions):
        identity=self.binding(inputs,positions);path=self.namespace/(self.digest(identity)+'.json')
        if path.is_symlink():raise ValueError('reference entry symlink forbidden')
        if not path.exists():return None
        payload=json.loads(path.read_text())
        if set(payload)!={'reference','record','record_sha256'} or payload['reference']!=self.identity:
            raise ValueError('reference identity differs')
        record=payload['record']
        if (set(record)!={'request_id','request_identity','prob','policy_id'} or
                self.digest(record)!=payload['record_sha256']):raise ValueError('reference record changed')
        bind_rollout(record,inputs,positions,policy_id=self.policy_id,group_size=2,seed=0)
        return record

    def write(self,record,inputs,positions):
        if set(record)!={'request_id','request_identity','prob','policy_id'}:
            raise ValueError('only label/action-free reference record accepted')
        bind_rollout(record,inputs,positions,policy_id=self.policy_id,group_size=2,seed=0)
        identity=self.binding(inputs,positions)
        payload=dict(reference=self.identity,record=record,record_sha256=self.digest(record))
        data=(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode()
        path=self.namespace/(self.digest(identity)+'.json')
        fd,temp=tempfile.mkstemp(prefix='.reference-',dir=self.namespace)
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
            try:os.link(temp,path);sync_directory(self.namespace)
            except FileExistsError:
                if self.read(inputs,positions)!=record:raise ValueError('non-deterministic fixed reference collision')
        finally:Path(temp).unlink(missing_ok=True)
        return path


class CachedReferenceEngine:
    """Read-only policy identity facade; cache misses require the fixed engine."""
    def __init__(self,cache,engine=None):
        if engine is not None and engine.bundle['policy_id']!=cache.policy_id:
            raise ValueError('fixed source engine identity differs')
        self.cache=cache;self.engine=engine;self.bundle={'policy_id':cache.policy_id}
        self.hits=0;self.misses=0

    def score(self,inputs,positions,*,audio_token_id):
        record=self.cache.read(inputs,positions)
        if record is not None:self.hits+=1;return record
        if self.engine is None:raise ValueError('fixed reference cache miss without source engine')
        if self.engine.bundle['policy_id']!=self.cache.policy_id:raise ValueError('reference engine changed')
        self.misses+=1
        record=self.engine.score(inputs,positions,audio_token_id=audio_token_id)
        self.cache.write(record,inputs,positions)
        return record
