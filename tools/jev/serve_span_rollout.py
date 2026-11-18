#!/usr/bin/env python3
"""Persistent isolated TP1 rollout engine over one local training connection."""
import argparse
import json
from pathlib import Path
import socket
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.vllm_spans import register_span_model
register_span_model()


def run(config, output):
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.local_runtime_storage import require_training_storage, check_paths
    from multimodal_jev.local_ipc import require_local_ipc
    from multimodal_jev.span_resident_rollout import ResidentSpanRollout
    from multimodal_jev.span_rollout_rpc import serve_connection
    require_training_storage(output);require_local_ipc()
    address = Path(config['socket_path']); exchange = Path(config['exchange'])
    check_paths({'socket':address.parent,'exchange':exchange})
    if len(str(address).encode()) > 107 or address.exists():
        raise ValueError('new short local socket path required')
    for path,digest in config['pins'].items():
        if file_hash(path) != digest:raise ValueError('rollout service dependency changed')
    policy = config['policy']
    processor = Qwen3ASRProcessor.from_pretrained(policy['model'],local_files_only=True)
    engine = ResidentSpanRollout(policy,policy['adapter'])
    listener = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        listener.bind(str(address));address.chmod(0o600);listener.listen(1)
        listener.settimeout(config.get('connect_timeout_seconds',900))
        atomic_json(output/'ready.json',dict(socket_path=str(address),base_id=policy['base_id'],
            runtime_profile=policy['runtime_profile'],physical_devices=policy['physical_devices'],
            scope='Engine initialized; caller must authenticate process and perform probability acceptance.'))
        connection,_ = listener.accept()
        with connection:
            connection.settimeout(config.get('request_timeout_seconds',1800))
            serve_connection(connection,engine,exchange,processor)
    finally:
        listener.close();engine.close()
        if address.exists():address.unlink()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from multimodal_jev.local_runtime_storage import require_training_storage
    require_training_storage(a.output)
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        run(json.loads(a.config.read_text()),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
