import json
from pathlib import Path
import tempfile
import threading
import time

import pytest

from multimodal_jev.baseline import file_hash
from tools.reports import serve_context_batched_candidate as service
from tools.reports.context_bulk_rpc_candidate import BulkContextEngineClient
from test_context_batched_service_candidate import engine_case
from test_context_bulk_rpc_candidate import record


@pytest.fixture
def environment(tmp_path, monkeypatch):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    import multimodal_jev.local_ipc as ipc
    monkeypatch.setattr(storage, 'require_training_storage', lambda p: None)
    monkeypatch.setattr(storage, 'check_paths', lambda p: None)
    monkeypatch.setattr(checkpoint, 'check_paths', lambda p: None)
    monkeypatch.setattr(ipc, 'require_local_ipc', lambda: None)
    engine, rows, calls = engine_case(); allocations = []
    engine.segmented_receipt = {'production_promoted': False}
    engine.closed = False
    def close(): engine.closed = True
    engine.close = close
    def allocate(config): allocations.append(config); return engine
    monkeypatch.setattr(service, 'owner_class', lambda root: allocate)
    with tempfile.TemporaryDirectory(prefix='jev-batch-') as short:
        output=tmp_path/'output';output.mkdir()
        own=str(Path(service.__file__).resolve())
        config=dict(transport_protocol=service.PROTOCOL,source_root='unused',
            socket_path=str(Path(short)/'s'),exchange=str(tmp_path/'exchange'),
            pins={own:file_hash(own)},policy=dict(candidate_runtime=service.PROFILE,physical_devices=[7]))
        yield config, output, engine, rows, calls, allocations


def test_real_service_socket_ready_receipt_and_cleanup(environment):
    config, output, engine, rows, calls, allocations=environment
    errors=[]
    def worker():
        try:service.run(config,output)
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    deadline=time.monotonic()+5
    while not (output/'ready.json').exists() and thread.is_alive() and time.monotonic()<deadline:
        time.sleep(.01)
    assert (output/'ready.json').exists(), errors
    ready=json.loads((output/'ready.json').read_text())
    assert ready['model_forward_mode']==service.PROFILE
    assert ready['max_batch_queries']==engine.max_batch_queries
    client=BulkContextEngineClient(config['socket_path'],config['exchange'],bundle=engine.bundle,timeout=5)
    try:
        assert client.score_many(rows,audio_token_id=98)==[record('a'*64,r) for r in rows]
    finally:
        client.close();thread.join(5)
    assert not errors and not thread.is_alive() and engine.closed
    assert calls==[[7,7]] and len(allocations)==1
    assert not Path(config['socket_path']).exists()
    assert json.loads((output/'last-batch.json').read_text())['encode_calls']==1


@pytest.mark.parametrize('fault',['protocol','runtime','pin','missing_pin','long_path','occupied'])
def test_rejection_precedes_model_allocation(environment,fault):
    config, output, engine, rows, calls, allocations=environment
    if fault=='protocol':config['transport_protocol']='wrong'
    if fault=='runtime':config['policy']['candidate_runtime']='wrong'
    if fault=='pin':config['pins']={k:'0'*64 for k in config['pins']}
    if fault=='missing_pin':config['pins']={}
    if fault=='long_path':config['socket_path']='/tmp/'+'s'*108
    if fault=='occupied':Path(config['socket_path']).write_text('preserve')
    with pytest.raises(ValueError):service.run(config,output)
    assert not allocations
    if fault=='occupied':assert Path(config['socket_path']).read_text()=='preserve'
