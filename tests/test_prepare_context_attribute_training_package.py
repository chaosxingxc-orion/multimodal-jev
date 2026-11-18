import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from tools.reports.prepare_context_attribute_training_package import seal_runtime,verify_tree,prepare,command,IMAGE
from tools.reports.context_attribute_execution_candidate import configurations

ROOT=Path(__file__).resolve().parents[1]
FORMAL=ROOT/'.local/decision-objective-research/context-rl-grpo-arithmetic-v2-package'


def config():
    c=json.loads((FORMAL/'config.json').read_text());s=json.loads((FORMAL/'service-config.json').read_text())
    root='/data/xingchao/multimodal-jev/simple-final-v1/context-attribute-package-test'
    c,s=configurations(c,s,root=root,source_root=root+'/source',dataset=root+'/dataset',
        schedule=root+'/attribute-inputs/q_credit',schedule_sha256='a'*64,arm='q_credit',algorithm='grpo_v1',pins={})
    return root,c,s


def test_seal_actual_source_and_detect_added_or_changed_files(tmp_path):
    target=tmp_path/'source';result=seal_runtime(ROOT,target)
    h=file_hash(target/'RUNTIME.json')
    assert verify_tree(target,'RUNTIME.json',h)==result
    assert result['image']==IMAGE and result['training_dispatch_allowed'] is False
    for name in ['tools/__init__.py','tools/reports/__init__.py','tools/jev/__init__.py']:
        assert (target/name).is_file()
    extra=target/'extra.py';extra.write_text('unexpected')
    with pytest.raises(ValueError):verify_tree(target,'RUNTIME.json',h)
    extra.unlink()
    (target/'tools/jev/serve_context_rollout.py').write_text('changed')
    with pytest.raises(ValueError):verify_tree(target,'RUNTIME.json',h)
    with pytest.raises(ValueError):seal_runtime(ROOT,target)


def test_missing_full_input_publication_prevents_package(tmp_path):
    runtime=tmp_path/'source';seal_runtime(ROOT,runtime)
    with pytest.raises(FileNotFoundError):prepare(runtime=runtime,runtime_sha256=file_hash(runtime/'RUNTIME.json'),
        formal=FORMAL,formal_sha256=file_hash(FORMAL/'PREPARED.json'),publication=tmp_path/'pending',
        publication_sha256='a'*64,dataset='unused',arm='q_credit',algorithm='grpo_v1',
        output=tmp_path/'new',remote_root='/data/xingchao/multimodal-jev/simple-final-v1/context-attribute-test')
    assert not (tmp_path/'new').exists()


def test_bounded_command_uses_explicit_devices_readonly_sources_and_local_writes():
    root,c,_=config()
    dependencies=['/mnt/cephfs/xingchao/experiments/frozen.py:/home/xingchao/frozen.py:ro']
    args=command(root,c,dependencies)
    devices=[args[i+1] for i,x in enumerate(args) if x=='--device']
    assert devices[:4]==[f'/dev/davinci{i}:/dev/davinci{i}' for i in (4,5,6,7)]
    mounts=[args[i+1] for i,x in enumerate(args) if x=='-v']
    assert '/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro' in mounts
    assert root+'/source:'+root+'/source:ro' in mounts
    assert root+'/exchange:/jev-exchange' in mounts
    assert all(not m.endswith(':rw') for m in dependencies)
    assert args[-5:]==[IMAGE,'python',root+'/source/tools/reports/context_attribute_execution_candidate.py','--package',root]
    assert '--volumes-from' not in args and '--network=none' in args
    with pytest.raises(ValueError):command(root,c,[dependencies[0][:-2]+'rw'])
    c['stop_update']=634
    with pytest.raises(ValueError):command(root,c,dependencies)
