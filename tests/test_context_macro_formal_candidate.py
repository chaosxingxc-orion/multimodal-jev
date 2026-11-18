import ast
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports import context_macro_formal_candidate as formal
from tools.reports.context_macro_trainer_candidate import experiment_window, generate as finite_source


def config():
    return dict(experiment_kind='formal_source', start_update=0, stop_update=634, stage='task',
        engineering_prefix=False, training_dispatch_allowed=True, checkpoint_interval=64,
        evaluation_steps=list(formal.STEPS), seed=317, algorithm=formal.ALGORITHM, dual_arm='joint')


def test_formal_contract_is_distinct_from_existing_engineering_and_efficiency():
    c=config()
    assert formal.contract(c)['recovery_checkpoint_steps']==[64,128,192,256,320,384,448,512,576,634]
    with pytest.raises(ValueError):experiment_window(c)
    for change in (dict(stop_update=100),dict(seed=318),dict(evaluation_steps=[0,634]),
                   dict(checkpoint_interval=1),dict(training_dispatch_allowed=False),
                   dict(engineering_prefix=True),dict(resume_root='/old'),dict(start_update=1),
                   dict(algorithm='grpo_v1'),dict(dual_arm='ranking_only'),dict(start_update=False)):
        with pytest.raises(ValueError):formal.contract(dict(c,**change))


def schedule():
    population=[str(i) for i in range(40513)]
    slots=[dict(id=uid,weight=1.,padding=False) for uid in population]
    slots += [dict(id=uid,weight=0.,padding=True) for uid in population[:63]]
    return [dict(epoch=e,index=i,slots=copy.deepcopy(slots[128*i:128*(i+1)]))
            for e in range(2) for i in range(317)], population


def test_complete_two_epochs_count_only_real_recordings():
    batches,population=schedule()
    counts=formal.validate_schedule(batches,population)
    assert sum(counts)==81026 and counts[316]==counts[633]==65
    assert counts.count(128)==632


@pytest.mark.parametrize('fault',['repeat_real','tail_weight','tail_flag','early_padding','cursor','missing_epoch'])
def test_corrupt_schedule_or_denominator_rejected(fault):
    batches,population=schedule()
    if fault=='repeat_real':batches[0]['slots'][1]['id']=batches[0]['slots'][0]['id']
    if fault=='tail_weight':batches[316]['slots'][-1]['weight']=1.
    if fault=='tail_flag':batches[316]['slots'][-1]['padding']=False
    if fault=='early_padding':batches[0]['slots'][0].update(weight=0.,padding=True)
    if fault=='cursor':batches[317]['index']=1
    if fault=='missing_epoch':batches.pop()
    with pytest.raises(ValueError):formal.validate_schedule(batches,population)


def test_generated_full_trainer_retains_native_update_dual_ack_and_short_trainer_bytes():
    before=finite_source(Path.cwd())
    derived=formal.generate(Path.cwd());ast.parse(derived)
    assert 'actor.update_policy(batch)' in derived
    assert 'session.accept_actor_update(update=index' in derived
    assert 'validate_schedule(batches,population.keys())' in derived
    assert "contract['formal_experiment']=formal" in derived
    assert "if not c['engineering_prefix']:" in derived  # processor admission retained
    assert 'formal_contract(c)' in derived and 'experiment_window(c)' not in derived
    assert finite_source(Path.cwd())==before


@pytest.mark.parametrize('step,checkpoint',[(1,False),(63,False),(64,True),(317,False),(633,False),(634,True)])
def test_actual_generated_export_section_publishes_every_policy_but_only_scheduled_recovery(tmp_path,step,checkpoint):
    # Execute the actual generated block with CPU tensors; FSDP/collectives and
    # serialization are substitutes. This does not prove NPU FSDP execution.
    model=torch.nn.Linear(2,1,bias=False)
    frozen=torch.nn.Parameter(torch.tensor([7.]),requires_grad=False)
    model.register_parameter('base_frozen',frozen)
    root=tmp_path;out=root/'rank-0';out.mkdir()
    commits={};captures=[];exports=[]
    def commit_file(path,writer):
        with path.open('wb') as stream:writer(stream)
    def commit_json(path,value):
        commits[path.name]=value;path.write_text(json.dumps(value))
    def capture(*args,**kwargs):captures.append(kwargs);return {'fixture_step':kwargs['completed_updates']}
    def export(source,target,tensors,**kwargs):
        assert set(tensors)=={'weight'} and tensors['weight'].dtype==torch.bfloat16
        exports.append(target.name);target.mkdir();(target/'MANIFEST.json').write_text('{}')
        return {'policy_id':f'p{step}'}
    def gather(peers,value):peers[:]=[value,copy.deepcopy(value)]
    def digest(path):return file_hash(path) if path.exists() else 'peer-shard-fixture'
    env=dict(index=step-1,formal=formal.contract(config()),wrapped=model,model=model,
        frozen={'base_frozen':tensor_identity(frozen)},tensor_identity=tensor_identity,
        named=list(model.named_parameters()),opt=None,rank=0,world=2,contract={'fixture':True},
        capture=capture,commit_file=commit_file,commit_json=commit_json,torch=torch,out=out,root=root,
        FSDP=SimpleNamespace(summon_full_params=lambda *a,**k:nullcontext()),
        dist=SimpleNamespace(all_gather_object=gather,barrier=lambda:None),
        canonical_execution=lambda value:value,export_checkpoint_policy=export,file_hash=digest,
        c=dict(source_policy='initial',source_policy_id='p0',source_contract={},base_id='base'))
    source=formal.generate(Path.cwd())
    block=source[source.index('                save_now=(index+1)'):source.index('                activation=[None]')]
    exec(compile(textwrap.dedent(block),'<actual-generated-checkpoint-block>','exec'),env)
    assert exports==[f'policy-{step:04d}']
    assert (root/f'execution-{step:04d}.pt').is_file()
    assert f'policy-{step:04d}-committed.json' in commits
    assert len(captures)==int(checkpoint) and env['save_now']==checkpoint
    assert (out/f'checkpoint-{step:04d}.pt').exists()==checkpoint
    assert (f'checkpoint-{step:04d}-committed.json' in commits)==checkpoint
    if checkpoint:assert len(commits[f'checkpoint-{step:04d}-committed.json']['rank_shards'])==2
