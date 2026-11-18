"""Real curve summarization with explicit source/NPU provenance substitutes."""
import copy
from pathlib import Path

import pytest

from multimodal_jev import local_checkpoint
from tools.reports import collect_context_replica_infonce_curve as module
from tools.reports.context_infonce_selection import STEPS
from test_context_selection_curve import config as experiment,points
from test_prepare_context_cached_training import write


def setup(tmp_path,monkeypatch):
    scores=points();exp=experiment();exp['evaluation']['checkpoints']=STEPS;exp['task_stage']['updates']=318
    plan=dict(source_audit={'fixture':'training'},experiment=exp,initial_policy_id='p0',
        reference_endpoint=dict(step=318,policy_id='p318'),checkpoints=[])
    entries=[];audits={}
    for index,(step,score) in enumerate(zip(STEPS,scores,strict=True)):
        score=copy.deepcopy(score);score['step']=step;score['result']['policy_id']=f'p{step}'
        score['result']['metrics']['query_macro_set_f1']=.8 if index==2 else .5
        root=tmp_path/f'package-{step}';write(root/'initial/policy/MANIFEST.json',{'policy':f'p{step}'})
        plan['checkpoints'].append(dict(step=step,policy_id=f'p{step}',manifest_sha256=module.file_hash(root/'initial/policy/MANIFEST.json')))
        m=dict(source_step=step,source_policy_id=f'p{step}',inference_services=[{'name':'a'},{'name':'b'}],
            layout={'services':{'a':'service-a.json','b':'service-b.json'}})
        write(root/'PREPARED.json',m)
        write(root/'parity-macro.json',dict(dual_runtime_identity={'same':'runtime'},contract={'same':'contract'},
            base_id='base',release_sha256=exp['release_sha256'],macro_queries=128,macro_client={'rollout_queries':32},
            audio_token_id=151676,cache_audit={'sha256':'cache'},cache_receipts={'sha256':'receipts'}))
        for name in ('a','b'):write(root/f'service-{name}.json',{'policy':{'policy_id':f'p{step}','max_batch_queries':32,'physical_devices':[6 if name=='a' else 7]}})
        write(root/'selection-inputs/panel.json',{'same':'panel'});(root/'selection-client.py').write_text('pass\n')
        spec=dict(output=str(root),source_step=step,parent={'same':'parent'},source_plan_sha256='plan',manifest_sha256=f'm{step}')
        bound=dict(selection_package=spec);path=tmp_path/f'audit-{step}.json';write(path,bound)
        entries.append(dict(step=step,audit_config=str(path),audit_config_sha256=module.file_hash(path)))
        audits[step]=dict(status='complete_replica_infonce_selection_audited',runtime_provenance_verified=True,
            actor_updates=0,parity={'exact':True},source_step=step,result=score['result'],manifest_sha256=f'm{step}',
            image='image',inference_audit={'input_receipts_sha256':'same'})
    for e in entries:write(Path(module.read(e['audit_config'])['selection_package']['output'])/'SOURCE-PLAN.json',plan)
    c=dict(source_plan='fixture-plan',source_plan_sha256='plan',evaluations=entries,output=str(tmp_path/'curve'))
    calls=[]
    monkeypatch.setattr(module,'reconstruct_plan',lambda c:copy.deepcopy(plan))
    def audit(bound,p):
        step=bound['selection_package']['source_step'];calls.append(step);return copy.deepcopy(audits[step])
    monkeypatch.setattr(module.point,'audit',audit)
    monkeypatch.setattr(module.publication,'check_paths',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    return c,plan,audits,calls


def test_complete_curve_keeps_selected_and_reference_roles_separate(tmp_path,monkeypatch):
    c,plan,_,calls=setup(tmp_path,monkeypatch);result=module.collect(c,{})
    assert calls==STEPS
    assert result['roles']['source_selected']==dict(step=64,policy_id='p64')
    assert result['roles']['stage2_initial_and_reference']==dict(step=318,policy_id='p318')
    assert result['points']==11 and result['target_inference_performed'] is False
    assert module.read(Path(c['output'])/'curve.json')['complete']


@pytest.mark.parametrize('fault',['missing','order','pin','policy','manifest','plan','runtime','client','panel',
    'service','input','threshold','population','parity','provenance','updates','point_audit'])
def test_partial_changed_or_unaudited_curve_cannot_publish(tmp_path,monkeypatch,fault):
    c,plan,audits,_=setup(tmp_path,monkeypatch);entry=c['evaluations'][1]
    root=Path(module.read(entry['audit_config'])['selection_package']['output']);proof=audits[32]
    if fault=='missing':c['evaluations'].pop()
    elif fault=='order':c['evaluations'].reverse()
    elif fault=='pin':entry['audit_config_sha256']='wrong'
    elif fault=='policy':proof['result']['policy_id']='other'
    elif fault=='manifest':write(root/'initial/policy/MANIFEST.json',{})
    elif fault=='plan':write(root/'SOURCE-PLAN.json',{})
    elif fault=='runtime':
        v=module.read(root/'parity-macro.json');v['dual_runtime_identity']={};write(root/'parity-macro.json',v)
    elif fault=='client':(root/'selection-client.py').write_text('different')
    elif fault=='panel':write(root/'selection-inputs/panel.json',{})
    elif fault=='service':write(root/'service-a.json',{'policy':{'max_batch_queries':4}})
    elif fault=='input':proof['inference_audit']['input_receipts_sha256']='changed'
    elif fault=='threshold':proof['result']['metrics']['threshold']=.6
    elif fault=='population':proof['result']['metrics']['queries']=9225
    elif fault=='parity':proof['parity']['exact']=False
    elif fault=='provenance':proof['runtime_provenance_verified']=False
    elif fault=='updates':proof['actor_updates']=1
    elif fault=='point_audit':
        def fail(*a):raise ValueError('actual collection corrupted')
        monkeypatch.setattr(module.point,'audit',fail)
    with pytest.raises(ValueError):module.collect(c,{})
    assert not Path(c['output']).exists()


def test_regression_is_retained_and_selects_initial_when_best(tmp_path,monkeypatch):
    c,_,audits,_=setup(tmp_path,monkeypatch)
    audits[64]['result']['metrics']['query_macro_set_f1']=.5
    audits[318]['result']['metrics']['query_macro_set_f1']=.4
    result=module.collect(c,{})
    assert result['roles']['source_selected']==dict(step=0,policy_id='p0')
    assert module.read(Path(c['output'])/'curve.json')['last_primary_delta']==pytest.approx(-.1)
