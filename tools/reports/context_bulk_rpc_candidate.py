"""Candidate bulk transport with unchanged sequential engine.score calls.

No tensor padding, parallel model forwards, cross-update behavior reuse, or
automatic fallback/retry. Separate from frozen production IPC implementation.
"""
import hashlib
import io
from pathlib import Path
import torch
from multimodal_jev.context_rollout_batch import bind_rollout
from multimodal_jev.local_checkpoint import commit_file
from multimodal_jev.span_rollout_rpc import send,receive
from tools.jev.context_rollout_rpc import ContextEngineClient

MAX_ROWS=128


def validate_rows(rows):
    if not isinstance(rows,list) or not 1<=len(rows)<=MAX_ROWS:
        raise ValueError('Bounded nonempty batch required')
    for row in rows:
        if (not isinstance(row,dict) or set(row)!={'inputs','positions'}
                or not isinstance(row['inputs'],dict)
                or set(row['inputs'])!={'input_ids','attention_mask','input_features','input_features_mask'}
                or any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' for v in [*row['inputs'].values(),row['positions']])):
            raise ValueError('Only label-free complete CPU tensors permitted')
    return rows


def read_rows(path,digest,exchange):
    path=Path(path).resolve();exchange=Path(exchange).resolve()
    if not path.is_relative_to(exchange) or path.suffix!='.pt':raise ValueError('Payload outside exchange')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Payload digest differs')
    return validate_rows(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True))


def serve_connection(connection,engine,exchange):
    sequence=0
    while True:
        try:command=receive(connection)
        except EOFError:return
        try:
            if type(command.get('sequence')) is not int or command['sequence']!=sequence:
                raise ValueError('Exact command sequence required')
            operation=command.get('operation')
            if operation=='score_many':
                if set(command)!={'sequence','operation','policy_id','path','sha256','audio_token_id'}:
                    raise ValueError('Label-free bulk command required')
                policy=command['policy_id']
                if engine.bundle['policy_id']!=policy:raise ValueError('Stale policy')
                rows=read_rows(command['path'],command['sha256'],exchange)
                result=[]
                for row in rows:
                    if engine.bundle['policy_id']!=policy:raise ValueError('Policy changed inside batch')
                    record=engine.score(**row,audio_token_id=command['audio_token_id'])
                    bind_rollout(record,**row,policy_id=policy,group_size=2,seed=0)
                    result.append(record)
                if engine.bundle['policy_id']!=policy:raise ValueError('Policy changed inside batch')
            elif operation=='switch':
                if set(command)!={'sequence','operation','previous_policy_id','snapshot','policy_id','adapter'}:
                    raise ValueError('Complete switch command required')
                if engine.bundle['policy_id']!=command['previous_policy_id']:raise ValueError('Stale switch source')
                result=engine.switch(**{k:command[k] for k in ('snapshot','policy_id','adapter')})
            elif operation=='close':
                if set(command)!={'sequence','operation'}:raise ValueError('Invalid close')
                send(connection,dict(sequence=sequence,ok=True,result=None));return
            else:raise ValueError('Unknown candidate operation')
            send(connection,dict(sequence=sequence,ok=True,result=result));sequence+=1
        except Exception as exc:
            send(connection,dict(sequence=sequence,ok=False,error=repr(exc)));raise


class BulkContextEngineClient(ContextEngineClient):
    def score_many(self,rows,*,audio_token_id):
        if self.phase!='ready':raise ValueError('Ready bulk owner required')
        path=self.exchange/f'bulk-{self.sequence:06d}.pt';policy=self.bundle['policy_id']
        try:
            validate_rows(rows)
            commit_file(path,lambda f:torch.save(rows,f))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            records=self._call('score_many',policy_id=policy,path=str(path),sha256=digest,audio_token_id=audio_token_id)
            if not isinstance(records,list) or len(records)!=len(rows):raise ValueError('Incomplete response batch')
            for row,record in zip(rows,records):bind_rollout(record,**row,policy_id=policy,group_size=2,seed=0)
            path.unlink()
            return records
        except BaseException:self.phase='failed';raise

    def score(self,inputs,positions,*,audio_token_id):
        return self.score_many([dict(inputs=inputs,positions=positions)],audio_token_id=audio_token_id)[0]
