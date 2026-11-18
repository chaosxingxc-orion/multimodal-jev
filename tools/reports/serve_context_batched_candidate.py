"""Separate finite-trial service; not a formal training entrypoint."""
import argparse
import json
import os
from pathlib import Path
import socket
import time

from multimodal_jev.baseline import atomic_json, file_hash
from tools.reports.context_batched_owner_candidate import owner_class, PROFILE
from tools.reports.context_batched_service_candidate import serve_connection

PROTOCOL = 'bulk_segmented_candidate_v1'


def run(config, output):
    from multimodal_jev.local_runtime_storage import require_training_storage, check_paths
    from multimodal_jev.local_ipc import require_local_ipc
    output = Path(output)
    if config.get('transport_protocol') != PROTOCOL or config['policy'].get('candidate_runtime') != PROFILE:
        raise ValueError('explicit segmented candidate service required')
    require_training_storage(output); require_local_ipc()
    address, exchange = Path(config['socket_path']), Path(config['exchange'])
    check_paths({'socket': address.parent, 'exchange': exchange})
    if not address.is_absolute() or len(os.fsencode(address)) > 107 or os.path.lexists(address):
        raise ValueError('new absolute short socket required')
    pins = config.get('pins')
    own = str(Path(__file__).resolve())
    if type(pins) is not dict or own not in pins:
        raise ValueError('pinned candidate entrypoint required')
    for name, digest in pins.items():
        if file_hash(name) != digest:
            raise ValueError('candidate dependency changed: ' + name)
    engine = owner_class(config['source_root'])(config['policy'])
    listener = None; bound = False
    try:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(address)); bound = True
        address.chmod(0o600); listener.listen(1); listener.settimeout(900)
        atomic_json(output/'ready.json', dict(policy_id=engine.bundle['policy_id'],
            contract=engine.bundle['contract'], physical_devices=config['policy']['physical_devices'],
            transport_protocol=PROTOCOL, model_forward_mode=PROFILE,
            max_batch_queries=engine.max_batch_queries, max_batch_tokens=engine.max_batch_tokens,
            segmented_receipt=engine.segmented_receipt, entrypoint_sha256=pins[own]))
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(1800)
            serve_connection(connection, engine, exchange)
        if hasattr(engine, 'last_batch_receipt'):
            atomic_json(output/'last-batch.json', engine.last_batch_receipt)
    finally:
        if listener is not None: listener.close()
        try: engine.close()
        finally:
            if bound and address.exists(): address.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(exist_ok=False)
    try:
        run(json.loads(args.config.read_text()), args.output)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))


if __name__ == '__main__':
    main()
