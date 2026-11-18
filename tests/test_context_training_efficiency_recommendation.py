import copy
import json
from pathlib import Path
import pytest
from tools.reports import context_training_efficiency_recommendation as r


def comparisons():
    def stats(total,warm):return dict(optimizer_steps=100,warmup_updates=2,
        all_updates=dict(updates=100,real_query_exposures=12800,complete_seconds=total),
        measured=dict(updates=98,real_query_exposures=12544,complete_seconds=warm))
    b,a,k=stats(100,98),stats(90,88),stats(80,78)
    def pair(schema,b,a,ids):return dict(schema=schema,baseline=b,candidate=a,
        matched_actual_queries=True,matched_reference_misses=True,
        source_audits=[dict(status='finite_dp2_efficiency_audited',files_sha256=i) for i in ids])
    return [pair('jev-acoustic-training-efficiency-v1',b,a,['b','a']),
            pair('jev-kv-training-efficiency-v1',a,k,['a','k'])]


def test_ranks_actual_training_time_and_never_promotes():
    result=r.choose(comparisons())
    assert result['fastest_observed_arm']=='acoustic_kv16'
    assert result['comparisons'][-1]['throughput_ratio']==1.25
    assert not result['automatic_promotion'] and not result['formal_training_dispatch_allowed']


@pytest.mark.parametrize('fault',['schema','partial','cache','bridge','manifest','nan','zero','bool','exposures'])
def test_refuses_invalid_or_unmatched_evidence(fault):
    c=copy.deepcopy(comparisons())
    if fault=='schema':c[0]['schema']='inference-only'
    elif fault=='partial':c[1]['candidate']['optimizer_steps']=99
    elif fault=='cache':c[1]['matched_reference_misses']=False
    elif fault=='bridge':
        c[1]['baseline']=copy.deepcopy(c[1]['baseline']);c[1]['baseline']['all_updates']['complete_seconds']=89
    elif fault=='manifest':c[1]['source_audits'][0]['files_sha256']='other'
    elif fault=='exposures':c[1]['candidate']['measured']['real_query_exposures']=12800
    else:c[1]['candidate']['all_updates']['complete_seconds']={'nan':float('nan'),'zero':0,'bool':True}[fault]
    with pytest.raises(ValueError):r.choose(c)


def test_disagreement_is_reported_and_ties_keep_simpler_runtime():
    c=comparisons();c[1]['candidate']['measured']['complete_seconds']=99
    out=r.choose(c);assert not out['window_rankings_agree'] and out['fastest_after_warmup_arm']=='acoustic'
    c=comparisons()
    for s in [c[0]['baseline'],c[0]['candidate'],c[1]['baseline'],c[1]['candidate']]:
        s['all_updates']['complete_seconds']=100;s['measured']['complete_seconds']=98
    assert r.choose(c)['fastest_observed_arm']=='service32'


@pytest.mark.parametrize('failed',[False,True])
def test_pending_or_failed_evidence_never_triggers_expensive_audit(tmp_path,monkeypatch,failed):
    def write(name,value):
        p=tmp_path/(name+'.json');p.write_text(json.dumps(value));return str(p)
    b={'manifest_sha256':'b'};a=dict(manifest_sha256='a',parent_operations=b);k=dict(manifest_sha256='k',parent_operations=a)
    ops={n:write(n,o) for n,o in zip(r.ARMS,(b,a,k))}
    studies={}
    for name,arm in [('acoustic','acoustic'),('kv','acoustic_kv16')]:
        root=tmp_path/(name+'-study');root.mkdir()
        studies[name]=write(name+'-config',dict(output=str(root),operations=ops[arm]))
        if failed and name=='acoustic':(root/'terminal.json').write_text('{"exit_code":1,"error":"failed"}')
    paths=[*ops.values(),*studies.values()]
    c=dict(operations=ops,studies=studies,pins={p:r.file_hash(p) for p in paths})
    monkeypatch.setattr(r,'audit_collected',lambda *a,**kw:pytest.fail('no partial audit'))
    out=r.report(c)
    assert out['status']==('failed_evidence_preserved' if failed else 'awaiting_complete_evidence')
    assert not out['formal_training_dispatch_allowed']


@pytest.mark.parametrize('fault',[None,'saved_report','audit_failure','missing_report','wrong_study'])
def test_completed_reports_are_reaudited_with_each_runtime(tmp_path,monkeypatch,fault):
    def save(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return str(p)
    b={'manifest_sha256':'b'};a=dict(manifest_sha256='a',parent_operations=b);k=dict(manifest_sha256='k',parent_operations=a)
    declared=dict(zip(r.ARMS,(b,a,k)))
    ops={n:save(n+'.json',o) for n,o in declared.items()}
    pairs=comparisons();studies={}
    for index,((name,status),arm) in enumerate(zip(r.STUDIES,('acoustic','acoustic_kv16'))):
        directory=tmp_path/(name+'-study')
        studies[name]=save(name+'-config.json',dict(output=str(directory),operations=ops[arm]))
        save(name+'-study/terminal.json',dict(exit_code=0,status=status))
        saved=copy.deepcopy(pairs[index])
        if fault=='saved_report' and name=='kv':saved['candidate']['all_updates']['complete_seconds']=1
        if fault!='missing_report' or name!='kv':save(name+'-study/comparison.json',saved)
    if fault=='wrong_study':
        save('kv-config.json',dict(output=str(tmp_path/'kv-study'),operations=ops['acoustic']))
    paths=[*ops.values(),*studies.values()]
    c=dict(operations=ops,studies=studies,pins={p:r.file_hash(p) for p in paths})
    events=[]
    audits={n:dict(status='finite_dp2_efficiency_audited',files_sha256=o['manifest_sha256']) for n,o in declared.items()}
    def audit(o,**kwargs):
        name=next(n for n,v in declared.items() if v==o);events.append('audit:'+name)
        module=dict(zip(r.ARMS,(r.baseline,r.acoustic,r.kv)))[name]
        assert kwargs['validator'] is module.validate
        assert kwargs['command_builder'] is module.command
        assert kwargs['output_auditor'] is r.acoustic.audit_outputs
        if fault=='audit_failure' and name=='acoustic':raise ValueError('raw audit failed')
        return audits[name]
    monkeypatch.setattr(r,'audit_collected',audit)
    def compare(left,right,proofs,index):
        names=r.ARMS[index:index+2]
        assert left==declared[names[0]] and right==declared[names[1]]
        assert proofs==[audits[n] for n in names]
        events.append('compare:'+str(index));return copy.deepcopy(pairs[index])
    monkeypatch.setattr(r,'acoustic_compare',lambda *a:compare(*a,0))
    monkeypatch.setattr(r,'kv_compare',lambda *a:compare(*a,1))
    if fault:
        with pytest.raises(ValueError):r.report(c)
    else:
        out=r.report(c)
        assert out['status']=='complete_efficiency_evidence_reconstructed'
        assert out['fastest_observed_arm']=='acoustic_kv16'
        assert len(out['evidence'])==4
        assert all(r.file_hash(p)==h for p,h in out['evidence'].items())
        assert not out['formal_training_dispatch_allowed']
    if fault in ('missing_report','wrong_study'):assert events==[]
    elif fault=='audit_failure':assert events==['audit:service32','audit:acoustic']
    else:assert events==['audit:service32','audit:acoustic','audit:acoustic_kv16','compare:0','compare:1']
