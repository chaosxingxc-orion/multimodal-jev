import json
from pathlib import Path

import pytest

from tools.reports import resume_context_selection_audit as recovery


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'experiment';point=root/'step-0064';point.mkdir(parents=True)
    def put(path,value):path.write_text(json.dumps(value));return recovery.sha(path)
    stopped=put(root/'status.json',dict(phase='stopped_preserve_handles',restart=False,error='audit_context_rl_selection_v2 failed'))
    put(root/'input.json',{})
    plan=dict(run_id='context-rl-test',algorithm='grpo_v1',evaluation_steps=[0,64,128])
    put(root/'plan.json',plan);monkeypatch.setattr(recovery,'plan',lambda spec:plan)
    prior=root/'step-0000';prior.mkdir();put(prior/'audit.json',dict(passed=True))
    collection=point/'collected';collection.mkdir();digest=put(collection/'record.json',{})
    transport=dict(files={'record.json':digest},container_id='id');put(point/'collected.transport.json',transport)
    accepted=tmp_path/'accepted.json'
    ah=put(accepted,dict(passed=True,step=64,run_id=plan['run_id'],algorithm='grpo_v1',**transport,
                        state=dict(Status='exited',ExitCode=0,OOMKilled=False)))
    config=dict(schema='selection-audit-recovery-v1',experiment_root=str(root),failed_status_sha256=stopped,
                accepted_audit=str(accepted),accepted_audit_sha256=ah,resume_step=64,output_directory=str(tmp_path/'recovery'))
    return root,point,config


def test_only_missing_audit_and_unstarted_suffix(tmp_path,monkeypatch):
    root,point,c=fixture(tmp_path,monkeypatch)
    assert recovery.validate(c)[0]==root
    (root/'step-0128').mkdir()
    with pytest.raises(ValueError,match='Later work'):recovery.validate(c)


@pytest.mark.parametrize('fault',['collection','proof','status','existing_audit','attempt'])
def test_rejects_unreconciled_or_changed_state(tmp_path,monkeypatch,fault):
    root,point,c=fixture(tmp_path,monkeypatch)
    paths={'collection':point/'collected/record.json','proof':tmp_path/'accepted.json',
           'status':root/'status.json','existing_audit':point/'audit.json',
           'attempt':tmp_path/'context-rl-eval-test-step0128-attempt.json'}
    paths[fault].write_text('{"changed":true}')
    with pytest.raises(ValueError):recovery.validate(c)


def test_resume_executes_only_missing_audit_then_remaining_point(tmp_path,monkeypatch):
    root,point,c=fixture(tmp_path,monkeypatch)
    spec=dict(experiment='experiment',stage1_root='stage1',stage1_proof='proof',template='template')
    (root/'input.json').write_text(json.dumps(spec))
    (root/'owner.lock').touch()
    for step in (0,64):
        (root/f'step-{step:04d}'/'audit-spec.json').write_text(json.dumps(dict(
            step=step,collection=f'collection-{step}',transport=f'transport-{step}',expected_config_sha256='digest')))
    monkeypatch.setattr(recovery,'load_config',lambda *args:(c,{}))
    monkeypatch.setattr(recovery,'check_paths',lambda *args:None)
    calls=[]
    def invoke(argv,**kwargs):
        calls.append(argv)
        def value(flag): return Path(argv[argv.index(flag)+1])
        if 'tools.reports.context_metric_python312' in argv:
            module=argv[argv.index('--module')+1]
            if module.endswith('audit_context_rl_curve_v2'):
                result=dict(complete=True)
            elif 'step-0064' in str(value('--output')):
                result=json.loads(Path(c['accepted_audit']).read_text())
            else: result=dict(passed=True)
            value('--output').write_text(json.dumps(result))
        elif 'tools.jev.prepare_context_rl_evaluation_package' in argv:
            package=value('--output');package.mkdir();(package/'PACKAGE.json').write_text('{}')
        elif 'tools.jev.launch_context_rl_evaluation_v2' in argv:
            (root.parent/'context-rl-eval-test-step0128-launch.json').write_text(json.dumps(dict(config_sha256='digest')))
        elif 'tools.jev.collect_context_rl_selection' in argv:
            value('--output').mkdir()
        else: raise AssertionError(argv)
    monkeypatch.setattr(recovery.subprocess,'run',invoke)
    recovery.run('template','local')
    assert len(calls)==6
    assert calls[0][calls[0].index('--module')+1]=='tools.jev.audit_context_rl_selection_v2'
    assert 'step-0064/audit-spec.json' in calls[0][calls[0].index('--spec')+1]
    assert sum('tools.jev.launch_context_rl_evaluation_v2' in call for call in calls)==1
    assert not any('train'==part for call in calls for part in call)
    assert json.loads((root/'status.json').read_text())['phase']=='complete_training_and_selection_curve'
    assert json.loads((Path(c['output_directory'])/'original-stopped-status.json').read_text())['restart'] is False


def test_relocation_gate_requires_matching_completed_alias(tmp_path):
    root=tmp_path/'experiment';root.mkdir()
    evidence=tmp_path/'evidence';evidence.mkdir()
    destination=tmp_path/'persistent';destination.mkdir()
    (destination/'collection').mkdir()
    collection=root/'training-collected';collection.symlink_to(destination/'collection')
    proof=tmp_path/'proof';proof.write_text('{}')
    intent=dict(collection=str(collection),owner_lock=str(root/'owner.lock'),destination=str(destination),
                transport=str(proof),proof=str(proof),transport_sha256=recovery.sha(proof),proof_sha256=recovery.sha(proof))
    (evidence/'intent.json').write_text(json.dumps(dict(config=intent)))
    config=dict(relocation_evidence=str(evidence),relocation_intent_sha256=recovery.sha(evidence/'intent.json'))
    with pytest.raises(FileNotFoundError):recovery.relocation_complete(config,root)
    (evidence/'COMPLETE.json').write_text(json.dumps(dict(status='relocated_verified_original_paths_preserved',
        source=str(collection),destination=str(destination))))
    recovery.relocation_complete(config,root)
    proof.write_text('{"changed":true}')
    with pytest.raises(ValueError):recovery.relocation_complete(config,root)


def test_recovery_rejects_changed_pins(tmp_path):
    source=tmp_path/'source';source.write_text('original')
    config=dict(schema='selection-audit-recovery-v2',pins={str(source):recovery.sha(source)})
    recovery.verify_recovery_pins(config)
    source.write_text('changed')
    with pytest.raises(ValueError):recovery.verify_recovery_pins(config)
