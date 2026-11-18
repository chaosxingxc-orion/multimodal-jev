"""Local Unix-socket transport for a persistent label-free Span rollout worker.

Only JSON controls cross the socket. Tensor payloads use weights_only loading
from an admitted local exchange directory, bound by SHA256. The owner supplies
isolated devices and authenticates the service/runtime before training.
"""
import hashlib
import json
from pathlib import Path
import socket
import struct

MAX_MESSAGE_BYTES = 64 * 1024**2


def send(sock, value):
    data = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_MESSAGE_BYTES: raise ValueError('rollout message exceeds bound')
    sock.sendall(struct.pack('!I', len(data))+data)


def receive(sock):
    def exact(size, *, clean_eof=False):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = sock.recv(size-len(chunks))
            if not chunk:
                if clean_eof and not chunks: raise EOFError('rollout peer closed between requests')
                raise ConnectionError('rollout peer closed inside a message')
            chunks.extend(chunk)
        return bytes(chunks)
    size = struct.unpack('!I', exact(4, clean_eof=True))[0]
    if size > MAX_MESSAGE_BYTES: raise ValueError('rollout message exceeds bound')
    return json.loads(exact(size))


def read_renderer(path, digest, exchange, processor):
    import torch
    path = Path(path).resolve(); exchange = Path(exchange).resolve()
    if not path.is_relative_to(exchange) or path.suffix != '.pt':
        raise ValueError('renderer must be inside local exchange directory')
    # Hash and deserialize the same bytes; no path reopen race.
    import io
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest: raise ValueError('renderer bytes changed')
    rows = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError('bounded complete renderer required')
    renderer = []
    for row in rows:
        if (set(row) != {'id','inputs','positions','candidates','request_id','identity'}
                or set(row['inputs'])-{'input_ids','attention_mask','position_ids','input_features','input_features_mask'}):
            raise ValueError('only label-free renderer fields permitted')
        renderer.append((dict(id=row['id']),row['inputs'],row['positions'],row['candidates'],
                         row['request_id'],row['identity'],processor))
    return renderer


class RolloutClient:
    """One owner, exact sequential requests, no retry after uncertain execution."""
    def __init__(self, socket_path, exchange, *, timeout=900.):
        from .local_runtime_storage import check_paths
        check_paths({'rollout_exchange': exchange, 'rollout_socket': Path(socket_path).parent})
        self.exchange = Path(exchange).resolve()
        self.exchange.mkdir(parents=True, exist_ok=True)
        self.socket = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        try:
            self.socket.connect(str(socket_path))
        except BaseException:
            self.socket.close()
            raise
        self.sequence = 0
        self.phase = 'ready'

    def rollout(self, config, compiled):
        import torch
        from .local_checkpoint import commit_file
        from .baseline import file_hash
        if self.phase != 'ready': raise ValueError('ready rollout connection required')
        self.phase = 'executing'
        try:
            rows = [dict(id=r[0]['id'],inputs=r[1],positions=r[2],candidates=r[3],
                         request_id=r[4],identity=r[5]) for r in compiled]
            path = self.exchange/f'renderer-{self.sequence:06d}.pt'
            commit_file(path,lambda stream:torch.save(rows,stream))
            send(self.socket,dict(sequence=self.sequence,operation='rollout',config=config,
                                  renderer=str(path),sha256=file_hash(path)))
            response = receive(self.socket)
            if response.get('sequence') != self.sequence or response.get('ok') is not True:
                raise ValueError('rollout worker failed or sequence differs: '+str(response.get('error')))
            result = response['result']
            if (result.get('passed') is not True or result.get('backend') != 'vllm'
                    or result.get('base_id') != config['base_id']
                    or result.get('adapter_id') != config['adapter_id']):
                raise ValueError('rollout worker returned wrong policy')
            # This ephemeral exchange payload is now consumed. Result/probability
            # evidence and checkpoint retention are owned by the caller.
            path.unlink()
            self.sequence += 1; self.phase = 'ready'
            return result
        except BaseException:
            self.phase = 'failed'
            raise

    def close(self):
        self.socket.close()
        self.phase = 'closed'


def serve_connection(connection, engine, exchange, processor):
    """Serve one training owner's stream; any failed request ends the session."""
    sequence = 0
    while True:
        try: command = receive(connection)
        except EOFError: return
        try:
            if (command.get('sequence') != sequence or type(command.get('sequence')) is not int
                    or command.get('operation') != 'rollout'
                    or set(command) != {'sequence','operation','config','renderer','sha256'}):
                raise ValueError('exact sequential rollout command required')
            rows = read_renderer(command['renderer'],command['sha256'],exchange,processor)
            result = engine.rollout(command['config'],rows)
            send(connection,dict(sequence=sequence,ok=True,result=result))
            sequence += 1
        except Exception as exc:
            send(connection,dict(sequence=sequence,ok=False,error=repr(exc)))
            raise
