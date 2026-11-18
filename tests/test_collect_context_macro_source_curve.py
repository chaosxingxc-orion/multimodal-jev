"""Full source curve producer; runtime/model audits are explicit test substitutes."""
import copy
import json
from pathlib import Path

import pytest

from test_context_selection_curve import config as experiment,points
from tools.reports import collect_context_macro_source_curve as module
from tools.reports.evaluate_context_transfer_macro import source_binding
from multimodal_jev import local_checkpoint


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path,monkeypatch):
    scores=points();scores[2]['result']['metrics']['query_macro_set_f1']=.8
    checkpoints=[dict(step=p['step'],policy_id=p['result']['policy_id']) for p in scores]
    parent=dict(manifest_sha256='parent')
    plan=dict(schema='jev-macro-source-evaluation-plan-v1',source_operations=parent,
        algorithm='dual_credit_grpo_floor_clipped_v1',experiment=experiment(),
        initial_policy_id='p0',checkpoints=checkpoints)
    operations={};audits={};entries=[]
    for checkpoint,point in zip(checkpoints,scores,strict=True):
        step=point['step'];key=str(step);root=tmp_path/f'point-{step}';package=root/'package'
        manifest=package/'policy/MANIFEST.json';write(manifest,{'policy_id':checkpoint['policy_id']})
        checkpoint['manifest_sha256']=module.file_hash(manifest)
        write(root/'PREPARED.json',dict(source_step=step,policy_id=checkpoint['policy_id'],parent_manifest_sha256='parent'))
        write(package/'evaluation.json',dict(macro_client=dict(runtime_identity={'profile':'same'}),
            contract={'test':True},base_id='base',release_sha256=plan['experiment']['release_sha256'],
            macro_queries=128,audio_token_id=151676,cache_audit={'sha256':'cache'},cache_receipts={'sha256':'receipts'}))
        write(package/'evaluation-inputs/input-receipts.json',[{'same':'evidence'}])
        for i in range(2):write(package/f'service-{i}.json',dict(policy=dict(policy_id=checkpoint['policy_id'],
            physical_devices=[6+i],max_batch_queries=8)))
        operations[key]=dict(prepared=str(root),manifest_sha256=f'manifest-{step}')
        audits[key]=dict(status='complete_source_point_macro_selection_audited',result=point['result'],
            files_sha256=f'manifest-{step}',image='same-image',transport_sha256=f'transport-{step}')
        entries.append(dict(step=step,config='operations',local=key))
    for ops in operations.values():write(Path(ops['prepared'])/'package/SOURCE-PLAN.json',plan)
    plan_path=tmp_path/'plan.json';write(plan_path,plan)
    config=dict(source_plan=str(plan_path),source_plan_sha256=module.file_hash(plan_path),
        evaluations=entries,output=str(tmp_path/'curve'))
    calls=[]
    monkeypatch.setattr(module,'load_config',lambda cfg,local:(operations[local],{}))
    def training(ops):assert ops==parent;calls.append('training');return {'actual':'training-audit'}
    def audit(ops):
        key=Path(ops['prepared']).name.split('-')[-1];calls.append(key);return audits[key]
    monkeypatch.setattr(module,'audit_training',training)
    monkeypatch.setattr(module,'build',lambda ops,admission:copy.deepcopy(plan))
    monkeypatch.setattr(module,'audit_point',audit)
    monkeypatch.setattr(module,'check_paths',lambda paths:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda paths:None)
    return config,plan,operations,audits,calls


def test_full_curve_freezes_roles_and_is_consumable_by_transfer_client(tmp_path,monkeypatch):
    c,plan,_,_,calls=setup(tmp_path,monkeypatch)
    result=module.collect(c,{'test':True})
    assert calls==['training',*[str(p['step']) for p in plan['checkpoints']]]
    assert result['selected_checkpoint']==dict(step=128,policy_id='p128')
    assert result['source_runtime_provenance_verified'] and not result['target_inference_performed']
    for role,identity,step in [('common_Stage1','p0',0),('source_selected_new_GRPO','p128',128),('fixed_endpoint_new_GRPO','p634',634)]:
        checked=source_binding(dict(source_evaluation=result['source_evaluation'],policy_role=role,policy_id=identity))
        assert checked['step']==step
    assert json.loads((Path(c['output'])/'terminal.json').read_text())=={'exit_code':0}


@pytest.mark.parametrize('fault',['missing','duplicate','reorder','step','policy','image','runtime','inputs',
    'population','threshold','plan','source_audit','point_audit','manifest','service_batch'])
def test_no_selection_from_incomplete_or_mismatched_curve(tmp_path,monkeypatch,fault):
    c,plan,ops,audits,_=setup(tmp_path,monkeypatch)
    if fault=='missing':c['evaluations'].pop()
    if fault=='duplicate':c['evaluations'][2]=c['evaluations'][1]
    if fault=='reorder':c['evaluations'].reverse()
    root=Path(ops['64']['prepared']);package=root/'package'
    if fault=='step':
        r=json.loads((root/'PREPARED.json').read_text());r['source_step']=128;write(root/'PREPARED.json',r)
    if fault=='policy':audits['64']['result']['policy_id']='other'
    if fault=='image':audits['64']['image']='other-image'
    if fault=='runtime':
        e=json.loads((package/'evaluation.json').read_text());e['macro_client']['runtime_identity']['profile']='other';write(package/'evaluation.json',e)
    if fault=='inputs':write(package/'evaluation-inputs/input-receipts.json',[])
    if fault=='population':audits['64']['result']['metrics']['queries']=9225
    if fault=='threshold':audits['64']['result']['metrics']['threshold']=.6
    if fault=='plan':write(package/'SOURCE-PLAN.json',dict(plan,initial_policy_id='other'))
    if fault in ('source_audit','point_audit'):
        def reject(*args):raise ValueError('actual audit failed')
        monkeypatch.setattr(module,'audit_training' if fault=='source_audit' else 'audit_point',reject)
    if fault=='manifest':write(package/'policy/MANIFEST.json',{'changed':True})
    if fault=='service_batch':write(package/'service-1.json',dict(policy=dict(physical_devices=[7],max_batch_queries=16)))
    with pytest.raises(ValueError):module.collect(c,{})
    assert not Path(c['output']).exists()


def test_no_gain_selects_step0_and_preserves_fixed_endpoint_regression(tmp_path,monkeypatch):
    c,_,_,audits,_=setup(tmp_path,monkeypatch)
    audits['128']['result']['metrics']['query_macro_set_f1']=.5
    audits['634']['result']['metrics']['query_macro_set_f1']=.4
    result=module.collect(c,{})
    assert result['selected_checkpoint']==dict(step=0,policy_id='p0')
    assert result['roles']['fixed_endpoint_new_GRPO']['policy_id']=='p634'
    curve=json.loads((Path(c['output'])/'curve.json').read_text())
    assert curve['last_primary_delta']==pytest.approx(-.1)


def test_source_score_tamper_is_rejected_by_target_role_binding(tmp_path,monkeypatch):
    c,_,_,_,_=setup(tmp_path,monkeypatch);result=module.collect(c,{})
    p=Path(c['output'])/'score-0128.json';score=json.loads(p.read_text())
    score['metrics']['query_macro_set_f1']=1.;write(p,score)
    with pytest.raises(ValueError,match='bound source evidence'):
        source_binding(dict(source_evaluation=result['source_evaluation'],policy_role='common_Stage1',policy_id='p0'))
