"""Label-free context engine IPC; one owner, no retry after uncertain execution.

The trainer keeps labels, rewards and action sampling. Only exact CPU inputs
and authenticated joint-policy paths reach the isolated vLLM process.
"""
import io
import hashlib
import socket
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_file
from multimodal_jev.span_rollout_rpc import send, receive
from multimodal_jev.context_rollout_batch import bind_rollout


def read_inputs(path, digest, exchange):
    path=Path(path).resolve();exchange=Path(exchange).resolve()
    if not path.is_relative_to(exchange) or path.suffix!='.pt':
        raise ValueError('input payload outside exchange')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('input bytes changed')
    value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if (not isinstance(value,dict) or set(value)!={'inputs','positions'} or
            not isinstance(value['inputs'],dict) or
            set(value['inputs'])!={'input_ids','attention_mask','input_features','input_features_mask'} or
            any(not isinstance(v,torch.Tensor) or v.device.type!='cpu'
                for v in [*value['inputs'].values(),value['positions']])):
        raise ValueError('only complete label-free CPU audio tensors permitted')
    return value


def serve_connection(connection,engine,exchange):
    sequence=0
    while True:
        try:command=receive(connection)
        except EOFError:return
        try:
            if type(command.get('sequence')) is not int or command['sequence']!=sequence:
                raise ValueError('exact command sequence required')
            operation=command.get('operation')
            if operation=='score':
                if set(command)!={'sequence','operation','policy_id','path','sha256','audio_token_id'}:
                    raise ValueError('label-free score command required')
                if engine.bundle['policy_id']!=command['policy_id']:raise ValueError('stale engine policy')
                value=read_inputs(command['path'],command['sha256'],exchange)
                result=engine.score(**value,audio_token_id=command['audio_token_id'])
            elif operation=='switch':
                if set(command)!={'sequence','operation','previous_policy_id','snapshot','policy_id','adapter'}:
                    raise ValueError('complete joint switch command required')
                if engine.bundle['policy_id']!=command['previous_policy_id']:raise ValueError('stale switch source')
                result=engine.switch(**{k:command[k] for k in ('snapshot','policy_id','adapter')})
            elif operation=='close':
                if set(command)!={'sequence','operation'}:raise ValueError('invalid close')
                send(connection,dict(sequence=sequence,ok=True,result=None));return
            else:raise ValueError('unknown context operation')
            send(connection,dict(sequence=sequence,ok=True,result=result))
            sequence+=1
        except Exception as exc:
            send(connection,dict(sequence=sequence,ok=False,error=repr(exc)))
            raise


class ContextEngineClient:
    def __init__(self,socket_path,exchange,*,bundle,timeout=900.):
        from multimodal_jev.local_runtime_storage import check_paths
        check_paths({'context_exchange':exchange,'context_socket':Path(socket_path).parent})
        self.exchange=Path(exchange).resolve();self.exchange.mkdir(parents=True,exist_ok=True)
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.socket.settimeout(timeout)
        try:self.socket.connect(str(socket_path))
        except BaseException:self.socket.close();raise
        self.bundle=bundle;self.sequence=0;self.phase='ready'

    def _call(self,operation,**fields):
        if self.phase!='ready':raise ValueError('ready context IPC owner required')
        self.phase='executing'
        try:
            send(self.socket,dict(sequence=self.sequence,operation=operation,**fields))
            response=receive(self.socket)
            if (response.get('sequence')!=self.sequence or response.get('ok') is not True):
                raise ValueError('context service rejected command: '+str(response.get('error')))
            self.sequence+=1;self.phase='ready'
            return response['result']
        except BaseException:self.phase='failed';raise

    def score(self,inputs,positions,*,audio_token_id):
        if self.phase!='ready':raise ValueError('ready context IPC owner required')
        path=self.exchange/f'inputs-{self.sequence:06d}.pt'
        try:
            commit_file(path,lambda f:torch.save(dict(inputs=inputs,positions=positions),f))
            result=self._call('score',policy_id=self.bundle['policy_id'],path=str(path),
                sha256=file_hash(path),audio_token_id=audio_token_id)
            bind_rollout(result,inputs,positions,policy_id=self.bundle['policy_id'],group_size=2,seed=0)
            path.unlink()
            return result
        except BaseException:self.phase='failed';raise

    def switch(self,*,snapshot,policy_id,adapter):
        # Authenticate on both sides. The worker separately verifies actual
        # loaded adapter/head tensors before acknowledging activation.
        from context_policy_activation import prepare_switch
        if self.phase!='ready':raise ValueError('ready context IPC owner required')
        try:
            bundle=prepare_switch(self.bundle,snapshot=snapshot,policy_id=policy_id,adapter=adapter,
                base_id=self.bundle['contract']['base_id'])
            result=self._call('switch',previous_policy_id=self.bundle['policy_id'],
                snapshot=str(snapshot),policy_id=policy_id,adapter=str(adapter))
            if (result.get('policy_id')!=policy_id or result.get('head_marker_exact') is not True or
                    result.get('native_loaded_tensors')!=112):raise ValueError('activation receipt differs')
            self.bundle=bundle
            return result
        except BaseException:self.phase='failed';raise

    def close(self):
        try:
            if self.phase=='ready':self._call('close')
        finally:self.socket.close();self.phase='closed'
