"""Actual service socket/handler lifecycle, substitute model and storage admission."""
import json
from pathlib import Path
import tempfile
import threading
import time

import pytest

from tools.reports import serve_context_transport_candidate as service
from tools.reports.context_bulk_rpc_candidate import BulkContextEngineClient
from context_rollout_rpc import ContextEngineClient
from multimodal_jev.baseline import file_hash
from test_context_bulk_rpc_candidate import row, record


@pytest.fixture
def environment(tmp_path, monkeypatch):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    import multimodal_jev.local_ipc as ipc
    import context_switch_rollout as switch
    monkeypatch.setattr(storage, 'require_training_storage', lambda p: None)
    monkeypatch.setattr(storage, 'check_paths', lambda p: None)
    monkeypatch.setattr(checkpoint, 'check_paths', lambda p: None)
    monkeypatch.setattr(ipc, 'require_local_ipc', lambda: None)
    engines = []

    class Engine:
        def __init__(self, config):
            self.bundle = dict(policy_id='a'*64, contract={'base_id':'b'*64})
            self.calls = []; self.closed = False
            engines.append(self)
        def score(self, inputs, positions, *, audio_token_id):
            self.calls.append(inputs['input_ids'].shape[1])
            return record(self.bundle['policy_id'], dict(inputs=inputs, positions=positions))
        def close(self):
            self.closed = True

    monkeypatch.setattr(switch, 'SwitchingContextRollout', Engine)
    with tempfile.TemporaryDirectory(prefix='jev-svc-') as short:
        output = tmp_path/'out'; output.mkdir()
        own = str(Path(service.__file__).resolve())
        config = dict(transport_protocol='bulk_sequential_v1', socket_path=str(Path(short)/'s'),
                      exchange=str(tmp_path/'exchange'), pins={own:file_hash(own)},
                      policy={'physical_devices':[7]})
        yield config, output, engines


@pytest.mark.parametrize('protocol,client_type', [
    ('sequential_v1', ContextEngineClient), ('bulk_sequential_v1', BulkContextEngineClient)])
def test_service_real_handler_and_cleanup(environment, protocol, client_type):
    config, output, engines = environment
    config['transport_protocol'] = protocol
    errors = []
    def worker():
        try: service.run(config, output)
        except BaseException as exc: errors.append(exc)
    thread = threading.Thread(target=worker, daemon=True); thread.start()
    deadline = time.monotonic()+5
    while not (output/'ready.json').exists() and thread.is_alive() and time.monotonic()<deadline:
        time.sleep(.01)
    assert (output/'ready.json').exists(), errors
    ready = json.loads((output/'ready.json').read_text())
    assert ready['transport_protocol'] == protocol
    rows = [row(3),row(7),row(3)]
    client = client_type(config['socket_path'], config['exchange'], bundle=engines[0].bundle, timeout=5)
    try:
        if protocol == 'sequential_v1':
            got = [client.score(**r,audio_token_id=1) for r in rows]
        else:
            got = client.score_many(rows,audio_token_id=1)
        assert got == [record('a'*64,r) for r in rows]
        assert client.sequence == (3 if protocol == 'sequential_v1' else 1)
    finally:
        client.close(); thread.join(5)
    assert not thread.is_alive() and not errors
    assert engines[0].calls == [3,7,3] and engines[0].closed
    assert not Path(config['socket_path']).exists()


@pytest.mark.parametrize('fault', ['protocol','pin','missing_pin','long_path','occupied'])
def test_reject_before_model_allocation(environment, fault):
    config, output, engines = environment
    if fault == 'protocol': config['transport_protocol']='unknown'
    if fault == 'pin': config['pins']={k:'0'*64 for k in config['pins']}
    if fault == 'missing_pin': config['pins']={}
    if fault == 'long_path': config['socket_path']='/tmp/'+'s'*108
    if fault == 'occupied': Path(config['socket_path']).write_text('preserve')
    with pytest.raises(ValueError): service.run(config, output)
    assert engines == []
    if fault == 'occupied': assert Path(config['socket_path']).read_text()=='preserve'
