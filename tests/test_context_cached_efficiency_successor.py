"""Controller tests; NPU execution/audits are explicitly substituted fixtures."""
from pathlib import Path

import pytest

from tools.reports import context_cached_efficiency_successor as module
from test_context_cached_acceptance_sequence import setup as sequence_setup
from test_context_reference_driver import write


def setup(tmp_path,monkeypatch):
    seq,refops,ref=sequence_setup(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    base=Path(seq['output']);driver=base.with_name(base.name+'-driver')/'started.json'
    write(base/'config.json',seq);write(driver,dict(pid=123,fixture=True))
    write(base/'terminal.json',dict(exit_code=0,status='cached_finite_sequence_audited',formal_training_dispatched=False))
    c=dict(output=str(tmp_path/'efficiency-successor'),efficiency_dispatch_allowed=True,
        predecessor=dict(config=str(base/'config.json'),driver=str(driver),terminal=str(base/'terminal.json')),
        experiment=dict(package=str(tmp_path/'efficiency-package'),operations_root=str(tmp_path/'efficiency-ops'),
            remote_root='/data/efficiency',tag='efficiency-test'))
    c['predecessor'].update({k+'_sha256':module.file_hash(c['predecessor'][k]) for k in ('config','driver')})
    evidence={}
    for name in module.sequence.STAGES:
        stage=seq['stages'][name];root=Path(stage['operations_root'])/'finalization'
        spec=dict(output=stage['package'],remote_root=stage['remote_root'],
            parent=refops['reference_job']['parent'],reference_job={'fixture':'producer'})
        if name=='recovery':spec['parent_audit']=evidence['prefix']
        ops=dict(stage,training_package=spec,reference_audit=ref,host=refops['host'],lease=refops['lease'],
            shared_root=refops['shared_root'],driver_root=refops['driver_root'],
            release_evidence=refops['release_evidence'])
        write(root/'config.json',ops)
        evidence[name]=module.sequence.audit_config(ops,ref)
        write(root/'audit-config.json',evidence[name])
        write(root/'terminal.json',dict(exit_code=0,status=module.sequence.STATUSES[name]))
    monkeypatch.setattr(module,'wait_exact',lambda *args:None)
    return c,seq,evidence


@pytest.mark.parametrize('fault',[None,'permission','pin','path','overlap','remote-overlap','keys','tag',
    'terminal','stage-terminal','deployment','host','audit-path','mutation'])
def test_predecessor_and_exact_stage_bindings(tmp_path,monkeypatch,fault):
    c,seq,evidence=setup(tmp_path,monkeypatch);p=c['predecessor']
    root=Path(seq['stages']['tail']['operations_root'])/'finalization'
    if fault=='permission':c['efficiency_dispatch_allowed']=False
    elif fault=='pin':p['config_sha256']='wrong'
    elif fault=='path':p['terminal']=str(tmp_path/'wrong.json')
    elif fault=='overlap':c['experiment']['package']=seq['stages']['prefix']['package']+'/child'
    elif fault=='remote-overlap':c['experiment']['remote_root']=seq['stages']['tail']['remote_root']
    elif fault=='keys':c['experiment']['stop_update']=101
    elif fault=='tag':c['experiment']['tag']='not safe'
    elif fault=='terminal':write(Path(p['terminal']),dict(exit_code=1))
    elif fault=='stage-terminal':write(root/'terminal.json',dict(exit_code=1))
    elif fault=='deployment':
        ops=module.read(root/'config.json');ops['package']+='-other';write(root/'config.json',ops)
    elif fault=='host':
        ops=module.read(root/'config.json');ops['host']='other';write(root/'config.json',ops)
        write(root/'audit-config.json',module.sequence.audit_config(ops,evidence['tail']['reference_audit']))
    elif fault=='audit-path':
        bound=module.read(root/'audit-config.json');bound['collection']+='-other';write(root/'audit-config.json',bound)
    elif fault=='mutation':
        monkeypatch.setattr(module,'wait_exact',lambda *args:write(Path(p['config']),dict(seq,changed=True)))
    if fault:
        with pytest.raises(ValueError):module.wait_acceptance(c)
    else:
        spec,ops,ref=module.wait_acceptance(c)
        assert spec['acceptance']==evidence and spec['acceptance_kind']=='efficiency'
        assert spec['parent']==evidence['prefix']['training_package']['parent']
        assert ops==evidence['prefix']['deployment'] and ref==evidence['prefix']['reference_audit']


@pytest.mark.parametrize('failure',[None,'predecessor','prepare','launch','collect','audit','status','cost'])
def test_single_attempt_order_stop_and_duplicate_rejection(tmp_path,monkeypatch,failure):
    c,seq,evidence=setup(tmp_path,monkeypatch);calls=[]
    if failure=='predecessor':write(Path(c['predecessor']['terminal']),dict(exit_code=1))
    def prepare(spec,p):
        calls.append('prepare')
        assert spec['acceptance']==evidence and spec['acceptance_kind']=='efficiency'
        if failure=='prepare':raise ValueError('independent finite gates failed')
        return dict(manifest_sha256='fixture-manifest')
    def launch(ops,p):
        calls.append('launch')
        assert ops['efficiency_dispatch_allowed'] and ops['host']==evidence['prefix']['deployment']['host']
        assert ops['release_evidence']==evidence['prefix']['deployment']['release_evidence']
        if failure=='launch':raise ValueError('launch failed')
        return dict(status='wrong' if failure=='collect' else 'collected_independent_cached_training_audit_required')
    def audit(bound,p):
        calls.append('audit')
        assert bound['training_package']['manifest_sha256']=='fixture-manifest'
        if failure=='audit':raise ValueError('audit failed')
        return dict(status='wrong' if failure=='status' else 'cached_100step_efficiency_outputs_audited',
            formal_training_admitted=False)
    def measure(c,p):
        calls.append('cost')
        if failure=='cost':raise ValueError('cost evidence failed')
        write(Path(c['output']),{'fixture':'not NPU measurements'})
    monkeypatch.setattr(module.efficiency,'package_action',prepare)
    monkeypatch.setattr(module.efficiency,'launch',launch);monkeypatch.setattr(module.efficiency,'audit',audit)
    monkeypatch.setattr(module,'cost_config',lambda bound,output:dict(output=str(output)))
    monkeypatch.setattr(module.cost,'run',measure)
    if failure:
        with pytest.raises(ValueError):module.run(c,{})
    else:module.run(c,{})
    expected=['prepare','launch','audit','cost']
    if failure=='predecessor':expected=[]
    elif failure in ('prepare','launch','audit','cost'):expected=expected[:expected.index(failure)+1]
    elif failure=='collect':expected=expected[:2]
    elif failure=='status':expected=expected[:3]
    assert calls==expected
    terminal=module.read(Path(c['output'])/'terminal.json')
    assert terminal['exit_code']==int(failure is not None) and terminal['restart'] is False
    if not failure:assert terminal['updates']==100 and terminal['formal_training_dispatched'] is False
    with pytest.raises(FileExistsError):module.run(c,{})
    assert calls==expected


def test_cost_inputs_are_bound_to_exact_collections(tmp_path):
    ref=dict(collection=str(tmp_path/'ref'))
    bound=dict(collection=str(tmp_path/'train'),reference_audit=ref)
    for name,record in (('train',bound),('ref',ref)):
        for key in ('inspect','launch','transport'):
            path=tmp_path/(name+'-'+key+'.json');write(path,{'fixture':key});record[key]=str(path)
    for path in ('train/config.json','ref/reference-build-config.json','ref/reference-build/result.json'):
        write(tmp_path/path,{'fixture':path})
    result=module.cost_config(bound,tmp_path/'cost.json')
    assert result['run_root']==str(tmp_path/'train/run')
    for key,value in result.items():
        if isinstance(value,dict):assert value['sha256']==module.file_hash(value['path'])
