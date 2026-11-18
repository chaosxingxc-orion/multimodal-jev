#!/usr/bin/env python3
"""Isolated resident context vLLM service for one actor controller."""
import argparse,json,socket,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(c,out):
    from multimodal_jev.local_runtime_storage import require_training_storage,check_paths
    from multimodal_jev.local_ipc import require_local_ipc
    from context_switch_rollout import SwitchingContextRollout
    from context_rollout_rpc import serve_connection
    require_training_storage(out);require_local_ipc()
    address=Path(c['socket_path']);exchange=Path(c['exchange'])
    check_paths({'socket':address.parent,'exchange':exchange})
    if len(str(address).encode())>107 or address.exists():raise ValueError('new short socket required')
    for path,digest in c['pins'].items():
        if file_hash(path)!=digest:raise ValueError('service source changed')
    engine=SwitchingContextRollout(c['policy'])
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        listener.bind(str(address));address.chmod(0o600);listener.listen(1)
        listener.settimeout(900)
        atomic_json(out/'ready.json',dict(policy_id=engine.bundle['policy_id'],
            contract=engine.bundle['contract'],physical_devices=c['policy']['physical_devices']))
        connection,_=listener.accept()
        with connection:
            connection.settimeout(1800)
            serve_connection(connection,engine,exchange)
    finally:
        listener.close();engine.close()
        if address.exists():address.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(json.loads(a.config.read_text()),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
