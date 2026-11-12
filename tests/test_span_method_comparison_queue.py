import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from multimodal_jev.baseline import file_hash


@pytest.mark.parametrize('fault',[None,'identity','launch','failed_evaluation','transport','source','existing','bad_result'])
def test_only_all_successful_exact_evaluations_allow_comparison(tmp_path,monkeypatch,fault):
    path=Path(__file__).resolve().parents[1]/'tools/jev/queue_span_method_comparison.py'
    spec=importlib.util.spec_from_file_location('comparison_queue',path)
    queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    source=tmp_path/'compare.py';source.write_text('fixed')
    result=tmp_path/'result.json';output=tmp_path/'run';output.mkdir()
    evaluations={}
    for method in ['sft','rl','conditional']:
        root=tmp_path/method;root.mkdir();launch=root/'launch.json'
        launch.write_text(json.dumps(dict(container_id=method,launch_exit_code=0)))
        evaluations[method]=dict(root=str(root),container_id=method,launch_sha256=file_hash(launch))
    config=dict(evaluations=evaluations,pins={str(source):file_hash(source)},comparator=str(source),result=str(result),repository=str(tmp_path))
    if fault=='launch':(tmp_path/'rl/launch.json').write_text('{}')
    if fault=='source':source.write_text('changed')
    if fault=='existing':result.write_text('{}')
    monkeypatch.setattr(queue.subprocess,'check_output',lambda argv,**kw:json.dumps([dict(Id='replacement' if fault=='identity' else argv[-1])]))
    calls=[]
    def run(argv,**kwargs):
        if 'wait' in argv:
            return SimpleNamespace(returncode=1 if fault=='transport' else 0,stdout='1' if fault=='failed_evaluation' else '0',stderr='')
        calls.append(argv);result.write_text(json.dumps(dict(passed=fault!='bad_result')))
    monkeypatch.setattr(queue.subprocess,'run',run)
    if fault:
        with pytest.raises(ValueError):queue.run(config,output)
        assert not (output/'completed.json').exists()
        if fault!='bad_result':assert not calls
    else:
        queue.run(config,output)
        assert len(calls)==1 and (output/'completed.json').exists()
