import copy
from pathlib import Path

import pytest

from tools.reports import collect_context_replica_transfer_scores as m
from test_collect_context_full_transfer_scores import fixture as original_fixture
from test_prepare_context_train_parity_job import write


def fixture(tmp_path, monkeypatch, dataset='FSC_native31', alias=False):
    c, events, proofs = original_fixture(tmp_path, monkeypatch, dataset, alias)
    entries={};seen={}
    for role,entry in c['entries'].items():
        old=entry['operations'];policy=entry['policy_id']
        if policy not in seen:
            package=Path(old['prepared'])/'package'
            e=m.shared.read(package/'evaluation.json');write(package/'parity-macro.json',e)
            write(package/'PREPARED.json',{'remote_root':'/remote/'+policy+'/package'})
            spec=dict(output=str(package),manifest_sha256=old['manifest_sha256'],source_admission=old['source_admission'])
            seen[policy]=dict(transfer_package=spec,collection=str(Path(old['operations_root'])/'collected'))
            proofs[policy]['status']='complete_replica_transfer_audited'
            proofs[policy]['manifest_sha256']=proofs[policy].pop('files_sha256')
        entries[role]=dict(policy_id=policy,operations=seen[policy])
    c['entries']=entries
    def audit(ops,provenance):
        policy=ops['transfer_package']['source_admission']['policy_id'];events.append('audit:'+policy)
        return copy.deepcopy(proofs[policy])
    monkeypatch.setattr(m.operations,'audit',audit)
    return c,events,proofs


@pytest.mark.parametrize('dataset',['FSC_native31','MINDS_native14','SLURP_clean_TEST60'])
def test_shared_native_scoring_after_all_replica_audits(tmp_path,monkeypatch,dataset):
    c,events,_=fixture(tmp_path,monkeypatch,dataset)
    result=m.score(c)
    assert events==['audit:p0','audit:p1','audit:p2','original','gold']
    assert result['runtime_provenance_verified'] and result['unique_policies']==3
    assert result['evaluation_kind']==('source_domain_held_out' if dataset=='SLURP_clean_TEST60' else 'cross_dataset_transfer')


def test_duplicate_roles_reuse_exact_outputs_and_keep_zero_delta(tmp_path,monkeypatch):
    c,events,_=fixture(tmp_path,monkeypatch,alias=True)
    result=m.score(c)
    assert events==['audit:p0','audit:p2','original','gold']
    assert result['unique_policies']==2
    assert all(v==0 for v in result['comparisons'][m.shared.ROLES[1]]['delta'].values())


@pytest.mark.parametrize('fault',['runtime','parity','source','inputs','prediction','policy','duplicate'])
def test_no_target_labels_when_admission_fails(tmp_path,monkeypatch,fault):
    c,events,proofs=fixture(tmp_path,monkeypatch,alias=fault=='duplicate')
    e=c['entries'][m.shared.ROLES[2]];ops=e['operations'];spec=ops['transfer_package']
    if fault=='runtime':proofs['p2']['runtime_provenance_verified']=False
    elif fault=='parity':proofs['p2']['parity']['exact']=False
    elif fault=='source':spec['source_admission']['source_curve_config']={'wrong':True}
    elif fault=='policy':e['policy_id']='wrong'
    elif fault=='inputs':
        p=Path(spec['output'])/'parity-macro.json';v=m.shared.read(p);v['portable_full_inputs']['manifest_sha256']='wrong';write(p,v)
    elif fault=='prediction':write(Path(ops['collection'])/'evaluation/inference/predictions.json',{})
    else:c['entries'][m.shared.ROLES[1]]['operations']=dict(c['entries'][m.shared.ROLES[0]]['operations'],extra=True)
    with pytest.raises(ValueError):m.score(c)
    assert 'gold' not in events


def test_scores_are_new_local_artifacts_not_overwritten(tmp_path,monkeypatch):
    from multimodal_jev import local_checkpoint
    c,_,_=fixture(tmp_path,monkeypatch)
    c['output']=str(tmp_path/'scores')
    monkeypatch.setattr(m.shared,'check_paths',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    result=m.collect(c,{'test':'CPU model substitutes'})
    assert m.shared.read(Path(c['output'])/'scores.json')==result
    with pytest.raises(FileExistsError):m.collect(c,{})
