import hashlib
import json
from pathlib import Path
import shutil
import pytest
from tools.reports.context_attribute_selection_processor import validate_package, derive


def package(tmp_path):
    source=Path('.local/decision-objective-research/slurp-native-attribute-selection-candidate-v1')
    for name in ('inputs.jsonl','catalog.json'):
        shutil.copy2(source/name,tmp_path/name)
    shutil.copy2(source/'MANIFEST.json',tmp_path/'candidate-manifest.json')
    shutil.copy2('/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/official-clean-intent-v1/selection.audio.jsonl',tmp_path/'audio.jsonl')
    return tmp_path


def seal(root):
    pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.name!='FILES.json'}
    (root/'FILES.json').write_text(json.dumps(pins))


def test_actual_frozen_selection_and_derived_modules(tmp_path):
    root=package(tmp_path);seal(root)
    assert validate_package(root)['recordings']==9226
    audit=derive(Path.cwd(),'audit');verify=derive(Path.cwd(),'verify')
    assert callable(audit.run) and callable(verify.verify)
    inputs=[json.loads(s) for s in (root/'inputs.jsonl').read_text().splitlines()]
    audio=[json.loads(s) for s in (root/'audio.jsonl').read_text().splitlines()]
    assert len(audit.validate_sources(inputs,audio,9226))==9226
    with pytest.raises(ValueError):audit.validate_sources(inputs,audio,40513)


@pytest.mark.parametrize('file',['inputs.jsonl','audio.jsonl','candidate-manifest.json'])
def test_resealed_wrong_population_is_rejected(tmp_path,file):
    root=package(tmp_path)
    p=root/file;p.write_text(p.read_text()+'\n')
    seal(root)
    with pytest.raises(ValueError):validate_package(root)
