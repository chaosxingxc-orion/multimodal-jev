"""Separate fixed-engine service entry for sequential versus bulk IPC trials.

One protocol per process, one controller, no retry or fallback. This candidate
uses the existing real SwitchingContextRollout; it never changes model batching.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import time

from multimodal_jev.baseline import atomic_json, file_hash

PROTOCOLS = ('sequential_v1', 'bulk_sequential_v1')


def run(config, output):
    from multimodal_jev.local_runtime_storage import require_training_storage, check_paths
    from multimodal_jev.local_ipc import require_local_ipc
    from context_switch_rollout import SwitchingContextRollout
    from context_rollout_rpc import serve_connection as sequential
    from tools.reports.context_bulk_rpc_candidate import serve_connection as bulk

    output = Path(output)
    protocol = config.get('transport_protocol')
    if protocol not in PROTOCOLS:
        raise ValueError('Explicit candidate transport protocol required')
    require_training_storage(output)
    require_local_ipc()
    address, exchange = Path(config['socket_path']), Path(config['exchange'])
    check_paths({'socket': address.parent, 'exchange': exchange})
    if not address.is_absolute() or len(os.fsencode(address)) > 107 or os.path.lexists(address):
        raise ValueError('New absolute short socket required')
    pins = config.get('pins')
    own = str(Path(__file__).resolve())
    if not isinstance(pins, dict) or own not in pins:
        raise ValueError('Candidate entrypoint must be pinned')
    for name, digest in pins.items():
        if file_hash(name) != digest:
            raise ValueError('Candidate service source changed: ' + name)
    handler = sequential if protocol == 'sequential_v1' else bulk
    # Engine allocation follows storage, protocol, path and pin validation.
    engine = SwitchingContextRollout(config['policy'])
    listener = None
    bound = False
    try:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(address)); bound = True
        address.chmod(0o600); listener.listen(1); listener.settimeout(900)
        atomic_json(output / 'ready.json', dict(
            policy_id=engine.bundle['policy_id'], contract=engine.bundle['contract'],
            physical_devices=config['policy']['physical_devices'],
            transport_protocol=protocol, model_forward_mode='sequential_single_row',
            entrypoint_sha256=pins[own]))
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(1800)
            handler(connection, engine, exchange)
    finally:
        if listener is not None:
            listener.close()
        try:
            engine.close()
        finally:
            if bound and address.exists():
                address.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        run(json.loads(args.config.read_text()), args.output)
    except BaseException as exc:
        atomic_json(args.output / 'terminal.json', dict(
            exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
    atomic_json(args.output / 'terminal.json', dict(exit_code=0, finished_unix=time.time()))


if __name__ == '__main__':
    main()
