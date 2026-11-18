import copy
import json
from pathlib import Path

import pytest

from test_context_transfer_macro_scoring import fixture as native_fixture
from tools.reports import collect_context_full_transfer_scores as m
from tools.reports import score_context_transfer_macro as native


def fixture(tmp_path,monkeypatch,dataset='FSC_native31',alias=False):
    original_entries,_=native_fixture(tmp_path,monkeypatch,dataset,alias=alias)
    admitted=native.load({})
    n=len(admitted['cache_receipts'])
    population=[dict(id=str(i),audio_sha256=f'a{i}',request_sha256=f'q{i}') for i in range(n)]
    rows=[dict(p,rendered_sha256='rendered') for p in population]
    admitted.update(dataset=dataset,population=population,input_receipts=admitted['cache_receipts'],
        processor_sha256='processor',source_rows=rows,source_release={'path':'release','sha256':'release'})
    events=[];entries={};evaluations={};proofs={};ops_by_policy={}
    for role in m.ROLES:
        old=original_entries[role];e=copy.deepcopy(old['evaluation']);policy=e['policy_id']
        if policy not in ops_by_policy:
            package=tmp_path/(policy+'-package')/'package';package.mkdir(parents=True)
            run=tmp_path/(policy+'-run');dest=run/'collected/evaluation/inference';dest.mkdir(parents=True)
            src=Path(old['collection'])/'inference/predictions.json'
            (dest/'predictions.json').write_bytes(src.read_bytes())
            e.update(snapshot='/remote/'+policy+'/package/policy',
                portable_full_inputs=dict(root='/remote/'+policy+'/package/full-inputs',
                    manifest_sha256='full',population_evidence_sha256='evidence'))
            (package/'evaluation.json').write_text(json.dumps(e));evaluations[policy]=e
            origin=dict(source_evaluation={'path':'source','sha256':'source'},
                source_curve_config={'plan':'source-plan'},policy_role=role,policy_id=policy)
            ops=dict(prepared=str(package.parent),operations_root=str(run),
                manifest_sha256='package-'+policy,source_admission=origin)
            ops_by_policy[policy]=ops
            proofs[policy]=dict(status='full_portable_transfer_outputs_audited',
                runtime_provenance_verified=True,parity={'exact':True},policy_id=policy,
                dataset=dataset,policy_role=role,files_sha256=ops['manifest_sha256'],
                inference_audit=dict(policy_id=policy,predictions_sha256=native.file_hash(src)))
        entries[role]=dict(operations=ops_by_policy[policy],policy_id=policy)
    monkeypatch.setattr(m,'source_binding',native.source_binding)
    def audit(ops):
        pid=ops['source_admission']['policy_id'];events.append('audit:'+pid)
        return copy.deepcopy(proofs[pid])
    def original(c):events.append('original');return admitted
    monkeypatch.setattr(m,'audit_collected',audit)
    monkeypatch.setattr(m,'load_portable',lambda _:copy.deepcopy(admitted))
    monkeypatch.setattr(m,'load_original',original)
    original_score=m.score_admitted
    def scoring(*args):events.append('gold');return original_score(*args)
    monkeypatch.setattr(m,'score_admitted',scoring)
    c=dict(entries=entries,original_evaluation=original_entries[m.ROLES[0]]['evaluation'],scoring={})
    return c,events,proofs


@pytest.mark.parametrize('dataset',['FSC_native31','MINDS_native14','SLURP_clean_TEST60'])
def test_full_native_population_scores_after_all_runtime_audits(tmp_path,monkeypatch,dataset):
    c,events,_=fixture(tmp_path,monkeypatch,dataset)
    result=m.score(c)
    assert events==['audit:p0','audit:p1','audit:p2','original','gold']
    assert result['runtime_provenance_verified'] and result['unique_policies']==3
    metrics='recording_weighted' if dataset=='MINDS_native14' else 'metrics'
    assert result['policies'][m.ROLES[1]][metrics]['top1_event_accuracy']==1
    assert .49<result['policies'][m.ROLES[2]][metrics]['top1_event_accuracy']<.51


def test_alias_roles_reuse_execution_and_keep_zero_delta(tmp_path,monkeypatch):
    c,events,_=fixture(tmp_path,monkeypatch,alias=True)
    result=m.score(c)
    assert events==['audit:p0','audit:p2','original','gold']
    assert result['unique_policies']==2
    assert all(v==0 for v in result['comparisons'][m.ROLES[1]]['delta'].values())


@pytest.mark.parametrize('fault',['missing','policy','runtime','source','inputs','parity','transport',
    'predictions','adapter','rendering','role','source_curve','duplicate_execution'])
def test_no_gold_before_all_admissions(tmp_path,monkeypatch,fault):
    c,events,proofs=fixture(tmp_path,monkeypatch,alias=fault=='duplicate_execution')
    entry=c['entries'][m.ROLES[2]];ops=entry['operations']
    p=Path(ops['prepared'])/'package/evaluation.json';e=json.loads(p.read_text())
    if fault=='missing':del c['entries'][m.ROLES[2]]
    if fault=='policy':entry['policy_id']='wrong'
    if fault=='runtime':e['macro_client']['runtime_identity']={'other':True}
    if fault=='source':ops['source_admission']['source_evaluation']['sha256']='other'
    if fault=='source_curve':ops['source_admission']['source_curve_config']={'other':True}
    if fault=='inputs':e['portable_full_inputs']['manifest_sha256']='other'
    if fault=='parity':proofs['p2']['parity']['exact']=False
    if fault=='transport':proofs['p2']['runtime_provenance_verified']=False
    if fault=='role':proofs['p2']['policy_role']='wrong'
    if fault=='predictions':
        (Path(ops['operations_root'])/'collected/evaluation/inference/predictions.json').write_text('{}')
    if fault in ('adapter','rendering'):
        original=m.load_original
        def bad(c):
            value=copy.deepcopy(original(c))
            if fault=='adapter':value['population'][0]['audio_sha256']='changed'
            else:value['source_rows'][0]['rendered_sha256']='changed'
            return value
        monkeypatch.setattr(m,'load_original',bad)
    if fault=='duplicate_execution':
        c['entries'][m.ROLES[1]]['operations']=dict(c['entries'][m.ROLES[0]]['operations'],extra=True)
    p.write_text(json.dumps(e))
    with pytest.raises(ValueError):m.score(c)
    assert 'gold' not in events


def test_relocated_portable_source_hashes_do_not_change_original_source_identity(tmp_path,monkeypatch):
    c,events,_=fixture(tmp_path,monkeypatch)
    for i,role in enumerate(m.ROLES):
        p=Path(c['entries'][role]['operations']['prepared'])/'package/evaluation.json'
        e=json.loads(p.read_text());e['source_evaluation']={'path':f'/remote/{i}/source','sha256':f'portable-{i}'}
        p.write_text(json.dumps(e))
    result=m.score(c)
    assert result['matched_identity']['source']=={'path':'source','sha256':'source'}
    assert events[-1]=='gold'


def test_collection_is_new_local_artifact_and_never_overwrites(tmp_path,monkeypatch):
    c,_,_=fixture(tmp_path,monkeypatch)
    c['output']=str(tmp_path/'scores')
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    result=m.collect(c,{'fixture':True})
    assert json.loads((Path(c['output'])/'scores.json').read_text())==result
    assert (Path(c['output'])/'config.json').is_file()
    with pytest.raises(FileExistsError):m.collect(c,{})
