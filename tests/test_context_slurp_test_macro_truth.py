"""Artifact binding test; native label reconstruction is tested independently."""
import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from tools.reports import context_slurp_test_macro_truth as module


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'inputs';root.mkdir()
    def write(path,value,lines=False):
        path.write_text(''.join(json.dumps(v)+'\n' for v in value) if lines else json.dumps(value))
        return dict(path=str(path),sha256=file_hash(path))
    candidates=[dict(id=str(i),description=f'c{i}') for i in range(60)]
    inputs=[dict(id=str(i)) for i in range(12991)]
    assets=dict(tasks=write(tmp_path/'tasks.json',dict(intent=dict(schema=dict(candidates=candidates)))))
    assets['input_rows']=write(root/'inputs.jsonl',inputs,True)
    assets['definitions']=write(root/'definitions.json',dict(candidates=[dict(id=c['id'],definition=c['description']) for c in candidates]))
    for key in ('native_requests','native_targets','clean_metadata','excluded','population','reference_predictions','reference_result'):
        assets[key]=write(tmp_path/(key+'.json'),[])
    manifest=dict(task_source_sha256=assets['tasks']['sha256'],source_files={e['path']:e['sha256'] for e in assets.values()})
    assets['input_manifest']=write(root/'MANIFEST.json',manifest)
    contract=dict(schema='slurp-context-clean-test-scoring-v1',threshold=.5,optimization_allowed=False,artifacts=assets)
    scoring=dict(contract=write(tmp_path/'scoring.json',contract))
    admitted=dict(candidate_ids=[c['id'] for c in candidates],source_rows=inputs)
    calls=[]
    def reconstruct(*args):calls.append(args);return {'fixture':'truth'}
    monkeypatch.setattr(module,'reconstruct_targets',reconstruct)
    return dict(inputs={'inputs':str(root)}),scoring,admitted,contract,calls,write


def test_frozen_sources_and_full_executed_population_bind_before_native_reconstruction(tmp_path,monkeypatch):
    c,s,a,_,calls,_=fixture(tmp_path,monkeypatch)
    rows,truth=module.load(c,s,a)
    assert rows==a['source_rows'] and truth=={'fixture':'truth'} and len(calls)==1


@pytest.mark.parametrize('fault',['target_bytes','ontology','input_rows','threshold','unbound_target','population'])
def test_changed_gold_or_executed_population_rejected(tmp_path,monkeypatch,fault):
    c,s,a,contract,calls,write=fixture(tmp_path,monkeypatch)
    assets=contract['artifacts']
    if fault=='target_bytes':Path(assets['native_targets']['path']).write_text('[1]')
    if fault=='ontology':a['candidate_ids'].reverse()
    if fault=='input_rows':Path(c['inputs']['inputs'],'inputs.jsonl').write_text('{}\n')
    if fault=='threshold':contract['threshold']=.4
    if fault=='unbound_target':assets['native_targets']=write(tmp_path/'different.json',[])
    if fault=='population':a['source_rows']=a['source_rows'][:-1]
    s['contract']=write(tmp_path/'scoring.json',contract)
    with pytest.raises(ValueError):module.load(c,s,a)
    assert not calls
