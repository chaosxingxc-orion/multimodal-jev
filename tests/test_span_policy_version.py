import importlib.util
import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash

spec=importlib.util.spec_from_file_location('span_policy_version',Path(__file__).parents[1]/'tools/jev/check_span_policy_version.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.mark.parametrize('passed',[True,False])
def test_version_check_pins_fresh_reference_and_gates_ready(tmp_path,monkeypatch,passed):
    calls=[]
    def process(argv,check):
        assert check
        output=Path(argv[argv.index('--output')+1]);output.mkdir()
        config=json.loads(Path(argv[argv.index('--config')+1]).read_text())
        calls.append(output.name)
        if output.name=='hf':
            result={'records':[{'id':'a'}]}
        else:
            reference=Path(config['hf_results'])
            assert reference==tmp_path/'hf/result.json'
            assert config['pins'][str(reference)]==file_hash(reference)
            result=dict(passed=passed,base_id='base',adapter_id='new',runtime_profile='fixture',records=[{}])
        (output/'result.json').write_text(json.dumps(result))
        (output/'terminal.json').write_text('{"exit_code":0}')
    monkeypatch.setattr(runner.subprocess,'run',process)
    if passed:
        runner.run({'pins':{}},tmp_path)
        assert json.loads((tmp_path/'READY.json').read_text())['adapter_id']=='new'
    else:
        with pytest.raises(ValueError,match='parity failed'):
            runner.run({'pins':{}},tmp_path)
        assert not (tmp_path/'READY.json').exists()
    assert calls==['hf','vllm']
