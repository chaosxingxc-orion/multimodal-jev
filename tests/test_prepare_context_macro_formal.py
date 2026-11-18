"""Real package sealing with synthetic efficiency/runtime/schedule evidence."""
import json
from pathlib import Path
import subprocess

import pytest

from test_prepare_context_macro_efficiency import setup as efficiency_setup
from tools.reports import prepare_context_macro_efficiency as efficiency
from tools.reports import prepare_context_macro_formal as formal


def test_generator_validation_imports_sealed_package_in_fresh_process(tmp_path):
    root=tmp_path/'package';module=root/formal.FORMAL_SOURCE
    module.parent.mkdir(parents=True)
    (root/'tools/__init__.py').write_text('')
    (root/'tools/reports/__init__.py').write_text('')
    module.write_text("def generate(root):\n    raise ValueError('sealed-package-sentinel')\n")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        formal.validate_generator(root)
    assert 'sealed-package-sentinel' in caught.value.stderr
    module.write_text("def generate(root):\n    return 'pass'\n")
    (root/formal.FORMAL_SOURCES[1]).write_text("def generate(root):\n    return 'pass'\n")
    formal.validate_generator(root)


def comparison():
    arms = {}
    for micro, seconds in ((8, 9000.), (16, 7000.), (32, 7500.)):
        arms['micro'+str(micro)] = dict(microbatch=micro, statistics=dict(optimizer_steps=100,
            all_updates=dict(updates=100, real_query_exposures=12800, complete_seconds=seconds)))
    return dict(schema='context-macro-efficiency-comparison-v1', independent_audits=True,
        matched_actual_queries=True, matched_reference_misses=True, warmup_updates=2, arms=arms)


def test_schedule_uses_local_sealed_paths_and_preserves_shared_release(tmp_path,monkeypatch):
    c=dict(source_root='/remote/package', release='/shared/release',release_sha256='release-hash',
        schedule='/remote/package/schedule',schedule_sha256='schedule-hash',schedule_release_sha256='legacy-hash')
    calls=[]
    def release(path, identity, split, **kw):
        calls.append((path,identity,split,kw));return [{'id':'sample'}],{},{}
    def schedule(path, identity, **kw):
        assert path==str(tmp_path.resolve()/'schedule')
        assert identity=='schedule-hash' and kw['release_sha256']=='legacy-hash'
        return ['batch'],{}
    monkeypatch.setattr(formal,'load_release',release)
    monkeypatch.setattr(formal,'load_schedule',schedule)
    monkeypatch.setattr(formal,'validate_schedule',lambda b,p:[65])
    assert formal.audit_schedule(c,tmp_path)['real_exposures']==65
    assert calls==[('/shared/release','release-hash','learning',{'purpose':'optimize'})]
    assert c['schedule']=='/remote/package/schedule'


def setup(tmp_path, monkeypatch):
    c, _, _ = efficiency_setup(tmp_path, monkeypatch)
    arms = efficiency.prepare(c, {})['arms']; result = comparison()
    ops = {name:json.loads(Path(spec['operations']).read_text()) for name,spec in arms.items()}
    for name, spec in arms.items():result['arms'][name]['audit_manifest_sha256'] = spec['manifest_sha256']
    compare_config = dict(arms={name:dict(config='operations',local=name) for name in arms})
    def load(config, local):
        return (compare_config if config == 'comparison' else ops[local]), {}
    monkeypatch.setattr(formal, 'load_config', load)
    monkeypatch.setattr(formal, 'compare', lambda c:result)
    monkeypatch.setattr(formal, 'audit_schedule', lambda c,p:dict(real_exposures=81026,updates=634,tail_real_queries=65))
    monkeypatch.setattr(formal, 'check_paths', lambda paths:None)
    monkeypatch.setattr(formal, 'validate_generator', lambda root:None)
    config = dict(comparison_config='comparison', comparison_local='local', project=str(Path.cwd()),
        output=str(tmp_path/'formal'), remote_root='/local/formal',
        operations_root=str(tmp_path/'formal-ops'), tag='formal')
    return config, ops, result


def test_selection_uses_all_updates_and_smallest_micro_for_exact_tie():
    c=comparison();assert formal.select_arm(c)=='micro16'
    c['arms']['micro8']['statistics']['all_updates']['complete_seconds']=7000.
    assert formal.select_arm(c)=='micro8'


@pytest.mark.parametrize('fault', ['missing_arm','audit','samples','cache','warmup','budget','nan','zero','capacity'])
def test_selection_rejects_incomplete_or_confounded_evidence(fault):
    c=comparison();arm=c['arms']['micro16']
    if fault=='missing_arm':c['arms'].pop('micro32')
    if fault=='audit':c['independent_audits']=False
    if fault=='samples':c['matched_actual_queries']=False
    if fault=='cache':c['matched_reference_misses']=False
    if fault=='warmup':c['warmup_updates']=0
    if fault=='budget':arm['statistics']['optimizer_steps']=99
    if fault=='nan':arm['statistics']['all_updates']['complete_seconds']=float('nan')
    if fault=='zero':arm['statistics']['all_updates']['complete_seconds']=0.
    if fault=='capacity':arm['microbatch']=8
    with pytest.raises(ValueError):formal.select_arm(c)


def test_sealed_formal_uses_selected_frozen_sources_and_initial_stage1(tmp_path, monkeypatch):
    c,ops,result=setup(tmp_path,monkeypatch)
    source=Path(ops['micro16']['prepared'])/'package'
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    receipt=formal.prepare(c,{'test':True});dest=Path(c['output'])/'package'
    t=json.loads((dest/'training.json').read_text());e=json.loads((dest/'FORMAL.json').read_text())
    assert receipt['selected_microbatch']==16 and receipt['formal_launch_admitted'] is False
    assert receipt['training_dispatch_allowed'] is False
    assert t['microbatch']==16 and t['start_update']==0 and t['stop_update']==634
    assert t['checkpoint_interval']==64 and t['seed']==317
    assert t['source_policy_id']==json.loads(before['training.json'])['source_policy_id']
    assert 'resume_root' not in t and not (dest/'resume').exists()
    previous=json.loads(before['RUNTIME-SOURCES.json']);current=json.loads((dest/'RUNTIME-SOURCES.json').read_text())
    assert {k:v for k,v in current.items() if k not in formal.FORMAL_SOURCES}==previous
    for entry in formal.FORMAL_SOURCES:
        assert (dest/entry).read_bytes()==Path(entry).read_bytes()
    assert e['source_changes']==list(formal.FORMAL_SOURCES)
    assert e['selected_arm']=='micro16' and e['comparison']==result
    assert e['selection_rule']==formal.SELECTION_RULE
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    operations=json.loads((Path(c['output'])/'operations.json').read_text())
    assert 'finite_training_dispatch_allowed' not in operations
    assert formal.validate(operations)['formal_evidence_sha256']==formal.file_hash(dest/'FORMAL.json')


@pytest.mark.parametrize('fault',['audit','schedule','source','existing_output','overlap_other_arm','storage'])
def test_failed_admission_does_not_create_formal_package(tmp_path,monkeypatch,fault):
    c,ops,result=setup(tmp_path,monkeypatch)
    def reject(*args):raise ValueError('unverified dependency')
    if fault=='audit':monkeypatch.setattr(formal,'compare',reject)
    if fault=='schedule':monkeypatch.setattr(formal,'audit_schedule',reject)
    if fault=='source':(Path(ops['micro16']['prepared'])/'package/model_code.py').write_text('changed')
    if fault=='existing_output':Path(c['output']).mkdir()
    if fault=='overlap_other_arm':c['remote_root']=ops['micro32']['remote_root']+'/child'
    if fault=='storage':monkeypatch.setattr(formal,'check_paths',reject)
    with pytest.raises(ValueError):formal.prepare(c,{})
    assert not (Path(c['output'])/'package').exists()
