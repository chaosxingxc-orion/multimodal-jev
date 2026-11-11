import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from multimodal_jev.baseline import file_hash

spec=importlib.util.spec_from_file_location('evaluation_queue',Path(__file__).resolve().parents[1]/'tools/jev/queue_span_transfer_evaluation.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


@pytest.mark.parametrize('fault',[None,'replaced','wait_failure','training_failure','changed_code'])
def test_only_successful_original_training_with_unchanged_code_dispatches(tmp_path,monkeypatch,fault):
    output=tmp_path/'run';output.mkdir();source=tmp_path/'launcher.py';source.write_text('pinned')
    evaluation=tmp_path/'evaluation';evaluation.mkdir()
    config=dict(container='original',container_id='id',pins={str(source):file_hash(source)},launcher=str(source),training='training',evaluation=str(evaluation),device=5,repository=str(tmp_path))
    if fault=='changed_code':source.write_text('changed')
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *a,**kw:json.dumps([{'Id':'replacement' if fault=='replaced' else 'id'}]))
    launches=[]
    def run(command,**kwargs):
        if command[:4]==['ssh','npu','docker','wait']:
            return SimpleNamespace(returncode=1 if fault=='wait_failure' else 0,stdout='1' if fault=='training_failure' else '0',stderr='')
        launches.append(command)
        (evaluation/'launch.json').write_text(json.dumps(dict(launch_exit_code=0,container_id='evaluation-id')))
    monkeypatch.setattr(queue.subprocess,'run',run)
    if fault:
        with pytest.raises(ValueError):queue.run(config,output)
        assert not launches and not (output/'dispatched.json').exists()
    else:
        queue.run(config,output)
        assert len(launches)==1 and '--launch' in launches[0]
        assert json.loads((output/'dispatched.json').read_text())['container_id']=='evaluation-id'
