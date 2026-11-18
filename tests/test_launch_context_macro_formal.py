import copy
import json
from pathlib import Path
import shutil

import pytest

from test_prepare_context_macro_formal import setup
from tools.reports import prepare_context_macro_formal as prepare
from tools.reports import launch_context_macro_formal as m


def fixture(tmp_path,monkeypatch,*,service_arm=None):
    config,ops,_=setup(tmp_path,monkeypatch)
    for o in ops.values():o['lease']=str(tmp_path/'device.lock')
    # The current admitted path is micro8, although the preparer also supports
    # future complete microbatch comparisons.
    comparison=prepare.compare({})
    comparison['arms']['micro8']['statistics']['all_updates']['complete_seconds']=6000.
    if service_arm is not None:
        from test_compare_context_service_batch import fixture as timers, compare
        timer_ops,audits=timers(tmp_path/'timer-evidence',2. if service_arm=='candidate' else .8)
        services=compare(*timer_ops,audits)
        values=list(prepare.efficiency_choice(config))
        values[2]=services;values[3]=service_arm;values[6]=prepare.SERVICE_SELECTION_RULE
        monkeypatch.setattr(prepare,'efficiency_choice',lambda _:tuple(values))
        monkeypatch.setattr(m,'efficiency_choice',lambda _:tuple(values))
    receipt=prepare.prepare(config,{})
    formal_ops=json.loads((Path(config['output'])/'operations.json').read_text())
    tail_package=tmp_path/'tail/package'
    shutil.copytree(Path(ops['micro8']['prepared'])/'package',tail_package)
    addition=tail_package/prepare.FORMAL_SOURCES[0]
    addition.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(config['project'])/prepare.FORMAL_SOURCES[0],addition)
    tail_ops=dict(ops['micro8'],prepared=str(tail_package.parent),manifest_sha256='tail',
                  operations_root=str(tmp_path/'tail-ops'))
    tail_t=json.loads((tail_package/'training.json').read_text())
    tail_t.update(experiment_kind='tail_admission',stop_update=2)
    tail_t['macro_client']['attempt_id']='tail-admission-v1'
    (tail_package/'training.json').write_text(json.dumps(tail_t))
    tail=dict(status='finite_dp2_tail_outputs_audited',files_sha256='tail',image=receipt['image'],
              tail_admission={'real_queries':[65,128]})
    monkeypatch.setattr(m,'audit_tail',lambda _:copy.deepcopy(tail))
    monkeypatch.setattr(m,'audit_schedule',prepare.audit_schedule)
    return dict(operations=formal_ops,preparation=config,tail_operations=tail_ops,
                formal_training_dispatch_allowed=True),tail


def test_admission_rebuilds_selection_preserves_assets_and_reports_recovery_limit(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch)
    r=m.admit(c)
    assert r['status']=='fresh_formal_launch_admitted' and r['selected_arm']=='micro8'
    assert r['schedule']['real_exposures']==81026
    assert not r['sparse_npu_recovery_verified'] and not r['automatic_resume_allowed']
    assert len(r['source_evaluation_steps'])==11


@pytest.mark.parametrize('selected',['baseline','candidate'])
def test_launch_preserves_selected_service_arm_including_candidate_regression(tmp_path,monkeypatch,selected):
    c,_=fixture(tmp_path,monkeypatch,service_arm=selected)
    r=m.admit(c)
    assert r['selected_arm']==selected
    assert r['selection_rule']==prepare.SERVICE_SELECTION_RULE


@pytest.mark.parametrize('fault',['selection','asset','code','service','external_pin','tail_status',
    'tail_counts','tail_code','schedule','lease','runtime_sources','missing_pins'])
def test_admission_rejects_changed_formal_or_tail_before_launch(tmp_path,monkeypatch,fault):
    c,tail=fixture(tmp_path,monkeypatch)
    package=Path(c['operations']['prepared'])/'package'
    # Let mutation reach independent parent comparison instead of stopping at
    # the generic manifest validator. Real production uses both checks.
    original_validate=m.formal.validate
    admitted=original_validate(c['operations'])
    monkeypatch.setattr(m.formal,'validate',lambda _:admitted)
    if fault=='selection':
        old=m.efficiency_choice
        def changed(config):
            values=list(old(config));values[3]='micro32';return values
        monkeypatch.setattr(m,'efficiency_choice',changed)
    if fault=='asset':(package/'new-asset.json').write_text('{}')
    if fault=='code':(package/'model_code.py').write_text('changed')
    if fault=='tail_code':(Path(c['tail_operations']['prepared'])/'package/model_code.py').write_text('changed')
    if fault=='service':
        p=package/'service-1.json';v=json.loads(p.read_text());v['extra']='changed';p.write_text(json.dumps(v))
    if fault=='external_pin':
        p=package/'training.json';v=json.loads(p.read_text());v['pins']['/outside/file']='changed';p.write_text(json.dumps(v))
    if fault=='tail_status':tail['status']='not_audited'
    if fault=='tail_counts':tail['tail_admission']['real_queries']=[128,128]
    if fault=='schedule':monkeypatch.setattr(m,'audit_schedule',lambda *a:dict(real_exposures=12800,updates=100))
    if fault=='lease':c['operations']['lease']='other'
    if fault=='runtime_sources':(package/'RUNTIME-SOURCES.json').write_text('{}')
    if fault=='missing_pins':
        p=package/'training.json';v=json.loads(p.read_text());v['pins']={};p.write_text(json.dumps(v))
    with pytest.raises(ValueError):m.admit(c)


def test_run_uses_exact_shared_lifecycle_and_records_admission(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch);calls=[]
    def lifecycle(ops,provenance,**kw):
        calls.append((ops,provenance,kw));return {'exit_code':0}
    monkeypatch.setattr(m,'run_services',lifecycle)
    assert m.run(c,{'caller':'fixture'})=={'exit_code':0}
    assert calls[0][0]['finite_inference_dispatch_allowed'] is True
    assert calls[0][1]['formal_launch_admission']['selected_arm']=='micro8'
    assert calls[0][2]==dict(validator=m.formal.validate,command_builder=m.formal.command,required_devices=(4,5,6,7))
    c['formal_training_dispatch_allowed']=False
    with pytest.raises(ValueError):m.run(c,{})
    assert len(calls)==1
