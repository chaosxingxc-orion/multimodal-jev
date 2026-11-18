"""Composition checks use a source-audit fixture, real traces and real replay.

These do not replace the full original auditor or establish NPU admission.
"""
import copy
import json
from pathlib import Path

import pytest
import torch

from tools.reports import audit_objective_training as module
from tests.test_objective_plugin import ROOT,binding,surrogate
from tests.test_objective_trace import update_fixture
from multimodal_jev.objective_trace import digest


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path,monkeypatch):
    package=tmp_path/'package';collection=tmp_path/'collection'
    objective=package/'objective-extension/extensions/stage2_objective.py'
    objective.parent.mkdir(parents=True)
    objective.write_bytes((ROOT/'extensions/stage2_objective.py').read_bytes())
    config=dict(objective_plugin=dict(binding(objective),path='/data/package/objective-extension/extensions/stage2_objective.py'),
        objective_trace=True,experiment_kind='engineering_prefix',start_update=0,stop_update=2,
        batch_config={'microbatch':1})
    write(package/'config.json',config);write(package/'PREPARED.json',{'remote_root':'/data/package'})
    plugin,ranks,docs=update_fixture()
    # This fixture has one row per rank; production derive is tested elsewhere.
    monkeypatch.setattr(module,'derive',lambda _:dict(dp_replicas=3,local_effective_batch=1))
    for update in range(2):
        groups=copy.deepcopy([g for rank in ranks for g in rank])
        # Generate independently valid step identities and signed records.
        from multimodal_jev.context_dual_credit import prepare
        for rank,g in enumerate(groups):
            b=g['dual_credit']['binding'];b['update']=update
            # Reuse the record unchanged at update zero. For update one,
            # reconstruct through the public sampler, not by editing hashes.
            if update:
                g['dual_credit']=prepare(torch.tensor(b['logits']),torch.tensor(b['probability']),
                    b['candidates'],torch.tensor(b['labels'],dtype=torch.bool),
                    policy_id=b['policy_id'],input_id=b['input_id'],
                    query_id=b['query_id'],update=update,algorithm=b['algorithm'],
                    group_size=b['group_size'],seed=b['seed'])
            g['dual_identity']['update']=update
            from multimodal_jev.objective_trace import unpack
            args=unpack(docs[rank]['rows'][0]['inputs']);args['identities']=[g['dual_identity']]
            args['records']=[g['dual_credit']];args['logits'].requires_grad_();rows=[]
            plugin.hybrid_loss(**args,trace_observer=rows.append)
            doc=dict(schema='jev-objective-update-traces-v1',update=update,rows=rows)
            write(collection/f'run/rank-{rank}/objective-traces/objective-{update:04d}.json',doc)
        torch.save(groups,collection/f'run/groups-{update:04d}.pt')
    monkeypatch.setattr(module.cached,'expected_contract',lambda c,m:dict(original=True))
    # The real cached wrapper must inject our contract into its replica auditor.
    def shared_output(package,collection,*,initial_policy):
        c=read(Path(package)/'config.json')
        return dict(world_size=3,updates=[{},{}],contract=expected_contract(c,[]))
    monkeypatch.setattr(module.cached.shared,'audit_outputs',shared_output)
    # Reference fixture calls the nested replica auditor as production does.
    def cached_output(package,collection,*,reference_root):
        auditor=FunctionType(shared.audit_outputs.__code__,dict(shared.audit_outputs.__globals__,expected_contract=expected_contract))
        return auditor(package,collection,initial_policy=Path(package)/'initial/policy')
    monkeypatch.setattr(module.cached,'audit_outputs',cached_output)
    return package,collection,plugin


def test_keeps_shared_contract_and_replays_all_ranks(tmp_path,monkeypatch,surrogate):
    p,c,plugin=setup(tmp_path,monkeypatch)
    original=module.cached.shared.audit_outputs
    result=module.audit_outputs(p,c,reference_root='fixture',loss_atol=0.)
    assert result['contract']==dict(original=True,objective_plugin=plugin.identity)
    assert len(result['objective_trace_audit']['updates'])==2
    assert not result['training_dispatch_allowed'] and not result['fresh_process_replay_verified']
    assert module.cached.shared.audit_outputs is original


def test_transport_wrapper_keeps_objective_spec_and_parent_reference(monkeypatch):
    spec=dict(output='candidate',parent=dict(reference_job={'collection':'reference'}))
    c=dict(training_package=spec,objective_loss_atol=0.)
    seen=[]
    def package_check(actual,prov):
        assert actual is spec;seen.append('package');return {'passed':True}
    def output_check(package,collection,*,reference_root,loss_atol):
        assert reference_root=='reference' and loss_atol==0.;seen.append('outputs')
        return {'verified':True}
    def wrapper(c,provenance):
        audit_package(c['training_package'],provenance)
        return audit_outputs(c['training_package']['output'],'collected',
            reference_root=c['training_package']['reference_job']['collection'])
    monkeypatch.setattr(module,'audit_package',package_check)
    monkeypatch.setattr(module,'audit_outputs',output_check)
    monkeypatch.setattr(module.cached,'audit',wrapper)
    assert module.audit(c,{})==dict(verified=True,formal_training_admitted=False)
    assert seen==['package','outputs'] and 'reference_job' not in spec


@pytest.mark.parametrize('fault',['reward','missing','extra','source_failure'])
def test_refuses_incomplete_or_rehashed_evidence(tmp_path,monkeypatch,surrogate,fault):
    p,c,_=setup(tmp_path,monkeypatch)
    path=c/'run/rank-0/objective-traces/objective-0000.json'
    if fault=='missing':path.unlink()
    elif fault=='extra':write(path.parent/'extra.json',{})
    elif fault=='reward':
        doc=json.loads(path.read_text());row=doc['rows'][0];row['loss']+=1
        row['sha256']=digest({k:v for k,v in row.items() if k!='sha256'});write(path,doc)
    else:
        def fail(*a,**kw):raise ValueError('original source audit failed')
        monkeypatch.setattr(module.cached.shared,'audit_outputs',fail)
    with pytest.raises((ValueError,FileNotFoundError)):
        module.audit_outputs(p,c,reference_root='fixture',loss_atol=0.)


# Globals resolved by the deliberately scoped fixture functions above.
from types import FunctionType
from tools.reports.audit_context_cached_training import shared,expected_contract,read
from tools.reports.audit_context_cached_training import audit_package,audit_outputs
