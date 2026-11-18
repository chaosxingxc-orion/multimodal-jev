"""Scorer rejects an unrelated success receipt before touching ground truth."""
import importlib.util
import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash

spec=importlib.util.spec_from_file_location('simple_scorer', Path(__file__).parents[1]/'tools/jev/score_slurp_simple_final.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('fault', ['container','config','input','policy','exit'])
def test_execution_receipt_mismatch_precedes_scoring(tmp_path, monkeypatch, fault):
    def write(name, value):
        path=tmp_path/name;path.write_text(json.dumps(value));return str(path)
    inputs=write('inputs.json',{})
    results=write('result.json',{})
    terminal=write('terminal.json',{'exit_code':0})
    inference={'inputs':inputs,'base_id':'b','adapter_id':'a','pins':{inputs:file_hash(inputs)}}
    if fault=='input':inference['inputs']='foreign'
    if fault=='policy':inference['adapter_id']='foreign'
    inference_path=write('config.json',inference)
    launch={'container_id':'actual','image':'frozen','config_sha256':file_hash(inference_path)}
    if fault=='config':launch['config_sha256']='foreign'
    launch_path=write('launch.json',launch)
    observed={'container_id':'actual','image':'frozen','state':{'Status':'exited','ExitCode':0}}
    if fault=='container':observed['container_id']='foreign'
    if fault=='exit':observed['state']['ExitCode']=1
    observed_path=write('observed.json',observed)
    config=dict(inputs=inputs,backend_results=results,terminal=terminal,inference_config=inference_path,
                launch=launch_path,observed=observed_path,policy_id='b:a')
    config['pins']={value:file_hash(value) for key,value in config.items() if key!='policy_id'}
    def forbidden(*args,**kwargs):raise AssertionError('must reject before loading gold')
    monkeypatch.setattr(module,'load',forbidden)
    with pytest.raises(ValueError,match='lineage'):module.run(config)
