import copy
from pathlib import Path

import pytest

from tools.reports import context_replica_transfer_sequence as m
from test_context_cached_grpo_evaluation_sequence import setup as previous_setup
from test_prepare_context_train_parity_job import write


def setup(tmp_path,monkeypatch):
    prior,formal,selection,ops=previous_setup(tmp_path,monkeypatch)
    base=Path(prior['output']);write(base/'config.json',prior)
    driver=base.with_name(base.name+'-driver')/'started.json';write(driver,{'pid':123})
    write(base/'terminal.json',dict(exit_code=0,status='cached_grpo_source_curve_collected_and_audited',
        points=11,target_inference_performed=False))
    c=dict(output=str(tmp_path/'transfer'),prepared_root=str(tmp_path/'transfer-prepared'),
        operations_root=str(tmp_path/'transfer-ops'),scores_root=str(tmp_path/'transfer-scores'),
        remote_root='/data/transfer-matrix',tag_prefix='transfer-matrix',transfer_dispatch_allowed=True,
        pins={},predecessor=dict(config=str(base/'config.json'),driver=str(driver),terminal=str(base/'terminal.json')),
        datasets={})
    for k in ('config','driver'):c['predecessor'][k+'_sha256']=m.file_hash(c['predecessor'][k])
    for dataset,slug in zip(m.DATASETS,m.SLUGS):
        specs={}
        for kind in ('full','parity'):
            root=tmp_path/(slug+'-'+kind);write(root/'MANIFEST.json',{})
            specs['portable_'+kind+'_inputs']=dict(root=str(root),manifest_sha256=m.file_hash(root/'MANIFEST.json'),
                cache_root='/data/caches/'+slug)
        c['datasets'][dataset]=dict(specs,original_evaluation=dict(dataset=dataset,source_release={'sha256':'v1'}),scoring={})
    write(base/'source-plan.json',{})
    entries=[]
    for point in prior['points']:
        p=Path(point['operations_root'])/'finalization/audit-config.json'
        write(p,dict(selection_package=dict(output=point['package'],remote_root=point['remote_root']),deployment={'tag':point['tag']}))
        entries.append(dict(step=point['step'],audit_config=str(p),audit_config_sha256=m.file_hash(p)))
    curve=dict(output=str(base/'curve'),source_plan=str(base/'source-plan.json'),source_plan_sha256=m.file_hash(base/'source-plan.json'),evaluations=entries)
    write(base/'curve-config.json',curve);write(base/'curve/source-evaluation.json',{})
    source=dict(path=str(base/'curve/source-evaluation.json'),sha256=m.file_hash(base/'curve/source-evaluation.json'))
    roles={role:dict(role=role,policy_id=policy) for role,policy in zip(m.ROLES,('p0','p1','p2'))}
    write(base/'curve/result.json',dict(status='complete_source_curve_audited_and_roles_frozen',source_evaluation=source,roles=roles))
    bound=dict(training_package={'parent':selection['point_template']['parent']},deployment=selection['deployment'])
    def admit(request):return dict(binding=roles[request['policy_role']],source_operations=bound,training={'fixture':True})
    monkeypatch.setattr(m,'admit',admit);monkeypatch.setattr(m,'wait_exact',lambda *a:None)
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    return c,prior,selection


@pytest.mark.parametrize('fault',[None,'permission','complex','pin','path','overlap','remote','input','cache','tag',
    'terminal','publication','curve','point','admission','mutation'])
def test_exact_source_and_simple_scope_gates(tmp_path,monkeypatch,fault):
    c,prior,selection=setup(tmp_path,monkeypatch);base=Path(prior['output'])
    if fault=='permission':c['transfer_dispatch_allowed']=False
    elif fault=='complex':c['datasets']['Audio2Tool']={}
    elif fault=='pin':c['predecessor']['config_sha256']='wrong'
    elif fault=='path':c['predecessor']['terminal']='/wrong'
    elif fault=='overlap':c['prepared_root']=selection['points'][0]['package']
    elif fault=='remote':c['remote_root']=prior['points'][0]['remote_root']
    elif fault=='input':c['datasets'][m.DATASETS[0]]['portable_full_inputs']['manifest_sha256']='wrong'
    elif fault=='cache':c['datasets'][m.DATASETS[0]]['portable_parity_inputs']['cache_root']='/other'
    elif fault=='tag':c['tag_prefix']='bad tag'
    elif fault=='terminal':write(base/'terminal.json',{'exit_code':1})
    elif fault=='publication':write(base/'curve/source-evaluation.json',{'changed':True})
    elif fault=='curve':
        p=base/'curve-config.json';v=m.read(p);v['evaluations'].pop();write(p,v)
    elif fault=='point':
        p=Path(prior['points'][0]['operations_root'])/'finalization/audit-config.json';write(p,{})
    elif fault=='admission':
        def refuse(*a):raise ValueError('actual source failed')
        monkeypatch.setattr(m,'admit',refuse)
    elif fault=='mutation':monkeypatch.setattr(m,'wait_exact',lambda *a:write(base/'config.json',{}))
    if fault:
        with pytest.raises(ValueError):m.wait_source(c)
    else:
        bindings,parent,deployment=m.wait_source(c)
        assert set(bindings)==set(m.ROLES) and parent==selection['point_template']['parent']
        assert deployment==selection['deployment']


@pytest.mark.parametrize('failure',[None,'source','prepare','run','audit','score'])
@pytest.mark.parametrize('alias',[False,True])
def test_all_unique_inference_before_gold_and_fail_closed(tmp_path,monkeypatch,failure,alias):
    c,_,selection=setup(tmp_path,monkeypatch);calls=[]
    def wait(c):
        calls.append('source')
        if failure=='source':raise ValueError('source failed')
        return ({r:dict(policy_role=r,policy_id=p) for r,p in zip(m.ROLES,('p0','p0' if alias else 'p1','p2'))},
            selection['point_template']['parent'],selection['deployment'])
    def prepare(spec,p):
        calls.append('prepare')
        if failure=='prepare':raise ValueError('invalid')
        return {'manifest_sha256':'fixture'}
    def run(ops,p):
        calls.append('run')
        if failure=='run':raise ValueError('parity failed')
        spec=ops['transfer_package']
        write(Path(ops['operations_root'])/'finalization/audit-config.json',dict(transfer_package=spec,deployment=ops))
        return dict(status='complete_replica_transfer_audited',runtime_provenance_verified=failure!='audit',
            dataset=spec['dataset'],policy_id=spec['source_admission']['policy_id'],target_scored=False)
    def score(spec,p):
        calls.append('gold')
        assert calls.count('run')==(6 if alias else 9)
        if failure=='score':raise ValueError('scoring failed')
        if alias:assert spec['entries'][m.ROLES[0]]['operations']==spec['entries'][m.ROLES[1]]['operations']
        result=dict(dataset=spec['original_evaluation']['dataset'],runtime_provenance_verified=True)
        write(Path(spec['output'])/'scores.json',result);return result
    monkeypatch.setattr(m,'wait_source',wait);monkeypatch.setattr(m.preparation,'prepare',prepare)
    monkeypatch.setattr(m.operations,'run',run);monkeypatch.setattr(m.scoring,'collect',score)
    if failure:
        with pytest.raises(ValueError):m.run(c,{})
    else:m.run(c,{})
    terminal=m.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==int(failure is not None) and terminal['restart'] is False
    if failure in ('source','prepare','run','audit'):assert 'gold' not in calls
    elif not failure:assert calls[-3:]==['gold']*3
    before=copy.deepcopy(calls)
    with pytest.raises(FileExistsError):m.run(c,{})
    assert before==calls
