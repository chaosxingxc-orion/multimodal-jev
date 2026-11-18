import json

import pytest

from test_context_macro_evaluation import setup
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_macro_rollout import digest
from tools.reports.context_macro_evaluation import evaluate
from tools.reports.audit_context_macro_evaluation import audit
from tools.reports import audit_context_macro_evaluation as module
from multimodal_jev.local_config import load_config


def fixture(tmp_path,monkeypatch):
    engine,population,encode,args,_=setup(tmp_path,monkeypatch)
    independent=[encode(asset)[1] for asset in population]
    evaluate(engine,population,encode,**args)
    kwargs=dict(population=population,candidate_ids=args['candidate_ids'],input_receipts=independent,
        policy_id=args['policy_id'],runtime_identity=args['runtime_identity'],macro_queries=args['macro_queries'])
    return args['output'],engine.root,kwargs


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value))


def refresh_output_hashes(root):
    result=read(root/'result.json')
    for b in result['batches']:b['sha256']=file_hash(root/b['path'])
    for name,key in [('population.json','population_sha256'),('predictions.json','predictions_sha256'),
                     ('input-receipts.json','input_receipts_sha256')]:
        result[key]=file_hash(root/name)
    write(root/'result.json',result)


def test_reconstructs_every_real_pool_rollout_and_short_tail(tmp_path,monkeypatch):
    root,journal,kwargs=fixture(tmp_path,monkeypatch)
    result=audit(root,journal,**kwargs)
    assert result['recordings']==7 and result['batches']==2


@pytest.mark.parametrize('fault',['missing_batch','failed','extra_batch','aggregate','receipt','population','policy'])
def test_rehashed_output_corruption_is_rejected(tmp_path,monkeypatch,fault):
    root,journal,kwargs=fixture(tmp_path,monkeypatch)
    if fault=='missing_batch':(root/'batch-00000001.json').unlink()
    if fault=='failed':write(root/'terminal.json',{'exit_code':1})
    if fault=='extra_batch':write(root/'batch-00000002.json',{})
    if fault=='aggregate':
        value=read(root/'predictions.json');value['records'].reverse();write(root/'predictions.json',value)
    if fault=='receipt':
        value=read(root/'input-receipts.json');value[0]['audio_sha256']='changed';write(root/'input-receipts.json',value)
    if fault=='population':
        value=read(root/'population.json');value['assets'][0]['request_sha256']='changed';write(root/'population.json',value)
    if fault=='policy':kwargs['policy_id']='wrong'
    if fault!='missing_batch':refresh_output_hashes(root)
    with pytest.raises((ValueError,FileNotFoundError)):audit(root,journal,**kwargs)


@pytest.mark.parametrize('fault',['duplicate_slot','foreign_replica','wrong_logits','wrong_inputs','missing_rollout','failure_marker'])
def test_rehashed_journal_corruption_is_rejected(tmp_path,monkeypatch,fault):
    root,journal,kwargs=fixture(tmp_path,monkeypatch)
    directory=journal/'evaluation-00000000'
    manifest=read(directory/'INPUT.json');complete=read(directory/'COMPLETE.json')
    path=directory/'rollout-0000.json';rollout=read(path)
    if fault=='duplicate_slot':
        manifest['jobs'][0]['indices']=[0,0];rollout['indices']=[0,0]
    if fault=='foreign_replica':
        manifest['jobs'][0]['replica_id']=2;rollout['replica_id']=2
    if fault=='wrong_logits':rollout['records'][0]['logits']=[10.]
    if fault=='wrong_inputs':manifest['inputs'][0]['input_ids']['sha256']='changed'
    if fault=='missing_rollout':path.unlink()
    else:write(path,rollout)
    if fault=='failure_marker':write(directory/'FAILED.json',{})
    write(directory/'INPUT.json',manifest)
    complete['input_sha256']=digest(manifest);complete['rollout_hashes']['rollout-0000']=digest(rollout)
    write(directory/'COMPLETE.json',complete)
    saved=read(root/'batch-00000000.json');saved['macro_receipt']=complete;write(root/'batch-00000000.json',saved)
    refresh_output_hashes(root)
    with pytest.raises((ValueError,FileNotFoundError)):audit(root,journal,**kwargs)


def test_independent_processor_receipt_required_even_with_consistent_run_hashes(tmp_path,monkeypatch):
    root,journal,kwargs=fixture(tmp_path,monkeypatch)
    kwargs['input_receipts'][0]['inputs']['input_features']['sha256']='different_independent_audio'
    with pytest.raises(ValueError,match='input audit'):audit(root,journal,**kwargs)


def test_macro_config_entry_binds_independent_files_and_preserves_prior_audit(tmp_path,monkeypatch):
    root,journal,kwargs=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'check_paths',lambda paths:None)
    config=dict(inference=str(root),journal=str(journal),output=str(tmp_path/'audit.json'),
                **{k:kwargs[k] for k in ('policy_id','runtime_identity','macro_queries')})
    for key in ('population','candidate_ids','input_receipts'):
        path=tmp_path/(key+'.json');write(path,kwargs[key])
        config[key]=dict(path=str(path),sha256=file_hash(path))
    template=tmp_path/'template.json';local=tmp_path/'local.json'
    write(template,{'spec':'${SPEC}'})
    write(local,{'schema':'jev-local-macros-v1','macros':{'SPEC':config}})
    resolved,provenance=load_config(template,local)
    report=module.run(resolved['spec'],provenance)
    assert report['recordings']==7
    with pytest.raises(FileExistsError):module.run(config,provenance)
    write(tmp_path/'population.json',[])
    with pytest.raises(ValueError,match='independent input artifact'):module.run(config,provenance)
