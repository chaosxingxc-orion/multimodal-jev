"""Independent finite DP2 training audit, with dual-replica journal membership.

Does not admit full training, recovery, gradient equivalence or model efficacy.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_macro_rollout import digest, plan_rollouts
from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
from multimodal_jev.local_config import load_config
from multimodal_jev.span_request_identity import tensor_identity
from tools.jev.audit_context_rl_training import update_targets, verify_checkpoint
from tools.reports.audit_context_dual_training import expected_contract as dp4_contract
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.context_dual_group_audit import audit as audit_groups
from tools.reports.context_macro_trainer_candidate import repartition, topology, experiment_window
from tools.reports.context_macro_training_operations import validate, command
from tools.reports.context_master_export_audit import audit as audit_masters


def read(path):
    return json.loads(Path(path).read_text())


def positive(v):
    return type(v) in (int,float) and math.isfinite(v) and v>0


def activation(receipt,policy):
    if (receipt['policy_id']!=policy or receipt['head_marker_exact'] is not True
            or receipt['native_loaded_tensors']!=112 or len(receipt['replica_receipts'])!=2):
        raise ValueError('both replicas must acknowledge full adapter and head activation')
    for r in receipt['replica_receipts']:
        if r['policy_id']!=policy or r['head_marker_exact'] is not True or r['native_loaded_tensors']!=112:
            raise ValueError('individual replica activation differs')


@lru_cache(maxsize=4096)
def prefix_mask_count(encoded):
    """Recover the valid length only when a binary prefix mask matches its hash."""
    identity=json.loads(encoded);shape=identity['shape']
    dtypes={'torch.int64':torch.int64,'torch.int32':torch.int32,'torch.bool':torch.bool,
            'torch.float32':torch.float32}
    if len(shape)!=2 or shape[0]!=1 or not 0<shape[1]<=32768 or identity['dtype'] not in dtypes:
        raise ValueError('bounded binary prefix mask required')
    dtype=dtypes[identity['dtype']];n=shape[1]
    zero=torch.zeros(1,dtype=dtype).view(torch.uint8).numpy().tobytes()
    one=torch.ones(1,dtype=dtype).view(torch.uint8).numpy().tobytes()
    raw=bytearray(one*n);width=len(one)
    for count in range(n,-1,-1):
        if hashlib.sha256(raw).hexdigest()==identity['sha256']:return count
        if count:raw[(count-1)*width:count*width]=zero
    raise ValueError('mask hash does not match any binary valid-prefix length')


def audit_journal(directory, records, *, policy, client):
    """Bind actual recorded results to label-free identities and whole-rollout owners."""
    directory=Path(directory);manifest=read(directory/'INPUT.json');complete=read(directory/'COMPLETE.json')
    if (directory/'FAILED.json').exists():raise ValueError('failed macro cannot be consumed')
    if not records or len(records)>client['max_queries']:raise ValueError('bounded complete macro required')
    identities=[];costs=[]
    for r in records:
        validate_logits_record(r);identity=r['request_identity']
        if identity['policy_id']!=policy or r['policy_id']!=policy or digest(identity)!=r['request_id']:
            raise ValueError('actual request identity or policy differs')
        pos=torch.tensor([identity['positions']],dtype=torch.long)
        identities.append(dict(identity['inputs'],positions=tensor_identity(pos)))
        costs.append(max(1,sum(prefix_mask_count(json.dumps(identity['inputs'][k],sort_keys=True))
            for k in ('attention_mask','input_features_mask'))))
    jobs=plan_rollouts(costs,client['rollout_queries'])
    if (manifest!=dict(batch_id=directory.name,policy_id=policy,runtime_identity=client['runtime_identity'],
            inputs=identities,expected_indices=list(range(len(records))),jobs=jobs)):
        raise ValueError('macro input identity or whole-rollout assignment differs')
    hashes={};ordered=[None]*len(records)
    for job in jobs:
        r=read(directory/(job['rollout_id']+'.json'))
        if (any(r[k]!=v for k,v in job.items()) or r['policy_id']!=policy
                or len(r['records'])!=len(job['indices']) or not positive(r['seconds'])):
            raise ValueError('rollout owner, coverage or timer differs')
        hashes[job['rollout_id']]=digest(r)
        for slot,record in zip(job['indices'],r['records'],strict=True):
            if ordered[slot] is not None or record!=records[slot]:raise ValueError('rollout membership differs')
            ordered[slot]=record
    expected=dict(batch_id=directory.name,policy_id=policy,status='complete_macro_inference',
        input_sha256=digest(manifest),rollout_hashes=hashes,completed_indices=list(range(len(records))),
        result_sha256=digest(ordered),actor_update_performed=False)
    if (any(complete[k]!=v for k,v in expected.items()) or not positive(complete['seconds'])
            or any(x is None for x in ordered)):
        raise ValueError('complete macro receipt differs')
    return dict(batch_id=directory.name,rows=len(records),policy_id=policy,seconds=complete['seconds'])


def expected_contract(c):
    contract=dp4_contract(c)
    contract.update(world_size=2,macro_layout=topology(c,2,['davinci4','davinci5']))
    if 'schedule_release_sha256' in c:contract['schedule_release_sha256']=c['schedule_release_sha256']
    return contract


REFERENCE_KEYS=('request_id','request_identity','prob','policy_id')


def reference_index(package,c):
    """Authenticate the retained fixed-reference index when the seal declares one.

    A retained index is the offline precompute of the same population; every
    journal reference record must reproduce it exactly, so no query may fall
    back to a live reference engine call.
    """
    if c.get('reference_condition')!='precomputed_retained':
        if c.get('reference_index') is not None or c.get('reference_index_sha256') is not None:
            raise ValueError('reference index requires the precomputed retained condition')
        return None
    package=Path(package);path=package/'REFERENCE-INDEX.json'
    if (c.get('reference_index')!=c['source_root']+'/REFERENCE-INDEX.json'
            or path.is_symlink() or not path.is_file()
            or file_hash(path)!=c.get('reference_index_sha256')):
        raise ValueError('authenticated retained reference index required')
    index=read(path)
    if (not isinstance(index,dict) or not index
            or any(not isinstance(k,str) or len(k)!=64 or set(k)-set('0123456789abcdef') for k in index)
            or any(not isinstance(v,str) or len(v)!=64 or set(v)-set('0123456789abcdef')
                   for v in index.values())):
        raise ValueError('hashed fixed reference index required')
    return index


def reference_transition(receipt, seen, *, current, reference, precomputed=None):
    missing=[]
    for q in receipt['queries']:
        r=q['reference_record'];key=r['request_id']
        if key in seen:
            if seen[key]!=r:raise ValueError('fixed reference changed across repeated input')
        elif precomputed is not None:
            # The retained cache must already hold every scheduled query; a live
            # engine call here would mean the precompute and run populations differ.
            if precomputed.get(key)!=digest({k:r[k] for k in REFERENCE_KEYS}):
                raise ValueError('precomputed fixed reference record differs')
            seen[key]=r
        else:missing.append(r);seen[key]=r
    if (receipt['reference_misses']!=len(missing)
            or receipt['bulk_reference_calls']!=int(bool(missing)) or receipt['bulk_behavior_calls']!=1):
        raise ValueError('reference cache coverage differs')
    expected=[reference,current] if missing and current!=reference else []
    if [r['policy_id'] for r in receipt['switches']]!=expected:
        raise ValueError('reference and behavior activation sequence differs')
    for r,policy in zip(receipt['switches'],expected,strict=True):activation(r,policy)
    return missing


def audit_outputs(package, collection, *, formal=False):
    from multimodal_jev.slurp_portable_release import load
    from multimodal_jev.context_training_schedule import load_schedule
    from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
    package,collection=Path(package),Path(collection);c=read(package/'training.json')
    run=collection/'run';client=c['macro_client'];runtime=c['dual_runtime_identity'];start=c['start_update']
    if formal:
        from tools.reports.context_macro_formal_candidate import contract as formal_contract,validate_schedule
        formal_plan=formal_contract(c)
        efficiency=False
    else:
        formal_plan=None
        efficiency=experiment_window(c)=='efficiency'
    stop=c['stop_update']
    if start not in (0,1):raise ValueError('initial prefix or one-step replay required')
    binding=read(package/'REPLAY.json') if start else None
    parent=None
    if start:
        parent=audit_collected(binding['parent_operations'])
        if (parent!=binding['parent_audit'] or parent['status']!='finite_dp2_training_audited'
                or parent['containers']['client']!=binding['parent_container']
                or parent['files_sha256']!=binding['parent_manifest_sha256']):
            raise ValueError('replay requires independently verified initial prefix')
    layout=topology(c,2,['davinci4','davinci5']);contract=expected_contract(c)
    if formal:contract['formal_experiment']=formal_plan
    if read(run/'result.json')!=dict(passed=True,start_update=start,stop_update=stop,schedule_updates=634,
            learning_population=40513,engineering_prefix=not formal):raise ValueError('complete configured result required')
    initial=read_execution_bundle(package/'policy',expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    initial_behavior=binding['initial_behavior_policy_id'] if start else c['source_policy_id']
    for i in range(2):
        service=read(package/f'service-{i}.json');ready=read(collection/f'service-{i}/ready.json')
        if (read(collection/f'service-{i}/terminal.json')['exit_code']!=0
                or ready['physical_devices']!=[6+i] or ready['policy_id']!=initial_behavior
                or ready['contract']!=c['source_contract']
                or ready['max_batch_queries']!=service['policy']['max_batch_queries']
                or ready['model_forward_mode']!=runtime['profile']):
            raise ValueError('normal matching service startup/completion required')
    rows,task,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
    population={r['id']:r for r in rows};candidates=[v['id'] for v in task['schema']['candidates']]
    batches,_=load_schedule(package/'schedule',c['schedule_sha256'],release_sha256=c.get('schedule_release_sha256',c['release_sha256']),population=population)
    if len(population)!=40513 or len(batches)!=634 or len(candidates)!=60:raise ValueError('full frozen TRAIN schedule required')
    real_counts=validate_schedule(batches,population.keys()) if formal else [128]*len(batches)
    # Policy export is per update; only recovery state follows the declared
    # cadence, with the final update always saved.
    interval=c['checkpoint_interval']
    checkpoint_steps=(formal_plan['recovery_checkpoint_steps'] if formal else
        [s for s in range(start+1,stop+1) if (s-start)%interval==0 or s==stop])
    if formal:
        for directory,pattern in [(run,'checkpoint-*-committed.json'),
                                   *[(run/f'rank-{r}','checkpoint-*.pt') for r in range(2)]]:
            expected={f'checkpoint-{s:04d}-committed.json' if directory==run else f'checkpoint-{s:04d}.pt'
                      for s in checkpoint_steps}
            if {p.name for p in directory.glob(pattern)}!=expected:
                raise ValueError('exact predeclared recovery checkpoints required')
        if {p.name for p in run.glob('policy-*-committed.json')}!={f'policy-{s:04d}-committed.json' for s in range(1,635)}:
            raise ValueError('every formal update must publish a policy receipt')
    else:
        expected={f'checkpoint-{s:04d}.pt' for s in checkpoint_steps}
        for rank in range(2):
            if {p.name for p in (run/f'rank-{rank}').glob('checkpoint-*.pt')}!=expected:
                raise ValueError('declared cadence must match published optimizer shards')
    journal=collection/'journal'/client['attempt_id'];sequence=0;macro_summaries=[];summaries=[]
    current=initial_behavior;prior=initial['tensors'];reference_seen={}
    precomputed=reference_index(package,c)
    if start:
        prior=read_execution_bundle(package/'resume/policy-0001',expected_policy_id=current,
            expected_contract=c['source_contract'],expected_base_id=c['base_id'])['tensors']
        for rank in range(2):
            if read(run/f'rank-{rank}/restored.json')!=dict(exact_masters=True,exact_adam=True,completed_updates=1):
                raise ValueError('exact all-rank FP32 master and Adam restoration required')
    for index in range(start,stop):
        step=index+1;receipt=read(run/f'rollout-{index:04d}.json');batch=read(run/f'batch-{index:04d}.json')
        if receipt['runtime_identity']!=runtime:raise ValueError('rollout runtime differs')
        records=[q['behavior_record'] for q in receipt['queries']]
        lengths=[(r['request_identity']['inputs']['input_ids']['shape'][-1],
                  r['request_identity']['inputs']['input_features']['shape'][-1]) for r in records]
        planned=dict(batches[index],rank_indices=repartition(lengths,batches[index]['rank_indices'],
            microbatch=c['microbatch'],world_size=2))
        targets=update_targets(batch,receipt,planned,population,task,update=index,
            policy_id=current,reference_id=c['source_policy_id'])
        if not formal and any(s['padding'] for s in planned['slots']):raise ValueError('finite prefix batches must be real queries')
        missing=reference_transition(receipt,reference_seen,current=current,
            reference=c['source_policy_id'],precomputed=precomputed)
        if missing:
            # Session caches only probabilities. Match the journal's actual logits
            # to every independently bound reference record before auditing it.
            d=journal/f'batch-{sequence:08d}';m=read(d/'INPUT.json');full=[None]*len(missing)
            for job in m['jobs']:
                for slot,r in zip(job['indices'],read(d/(job['rollout_id']+'.json'))['records'],strict=True):full[slot]=r
            keys=('request_id','request_identity','prob','policy_id')
            if [{k:r[k] for k in keys} for r in full]!=missing:raise ValueError('reference macro membership differs')
            macro_summaries.append(audit_journal(d,full,policy=c['source_policy_id'],client=client));sequence+=1
        macro_summaries.append(audit_journal(journal/f'batch-{sequence:08d}',records,policy=current,client=client));sequence+=1
        groups=torch.load(run/f'groups-{index:04d}.pt',map_location='cpu',weights_only=True)
        if [g['request_id'] for g in groups]!=[batch['records'][i]['request_id'] for rank in planned['rank_indices'] for i in rank]:
            raise ValueError('actual DP2 group order differs')
        summary=audit_groups(groups,receipt['queries'],targets,candidates=candidates,algorithm=c['algorithm'],
            arm=c['dual_arm'],update=index,policy_id=current,reference_policy_id=c['source_policy_id'])
        save_now=step in checkpoint_steps
        commit=read(run/f'{"policy" if formal else "checkpoint"}-{step:04d}-committed.json')
        if save_now:
            checkpoint=read(run/f'checkpoint-{step:04d}-committed.json')
            shards={f'rank-{r}/checkpoint-{step:04d}.pt':file_hash(run/f'rank-{r}/checkpoint-{step:04d}.pt') for r in range(2)}
            if (checkpoint['rank_shards']!=shards
                    or formal and checkpoint!=dict(commit,rank_shards=shards)):
                raise ValueError('committed recovery shards differ from published policy')
        if (commit['updates']!=step or commit['contract']!=contract
                or commit['execution_file']!=f'execution-{step:04d}.pt'
                or commit['execution_sha256']!=file_hash(run/f'execution-{step:04d}.pt')
                or commit['joint_policy_directory']!=f'policy-{step:04d}'
                or commit['joint_policy_manifest_sha256']!=file_hash(run/f'policy-{step:04d}/MANIFEST.json')):
            raise ValueError('committed DP2 checkpoint identity differs')
        current=commit['joint_policy']['policy_id']
        bundle=read_execution_bundle(run/f'policy-{step:04d}',expected_policy_id=current,
            expected_contract=c['source_contract'],expected_base_id=c['base_id'])
        verify_native_adapter(run/f'adapter-{step:04d}',bundle)
        execution=torch.load(run/f'execution-{step:04d}.pt',map_location='cpu',weights_only=True)
        if (len(execution)!=128 or set(bundle['tensors'])!=set(execution)|{'decision_token_row'}
                or any(v.dtype!=torch.bfloat16 or not torch.equal(v,bundle['tensors'][k]) for k,v in execution.items())):
            raise ValueError('every update export must match the published BF16 policy')
        states=None;master=None
        if save_now:
            verify_checkpoint(run,step,contract,bundle,world_size=2)
            states=[torch.load(run/f'rank-{r}/checkpoint-{step:04d}.pt',map_location='cpu',weights_only=True) for r in range(2)]
            master=audit_masters(states,execution,world_size=2)
        if not torch.equal(bundle['tensors']['decision_token_row'],initial['tensors']['decision_token_row']):raise ValueError('frozen marker changed')
        changes={kind:sum(not torch.equal(v,prior[k]) for k,v in bundle['tensors'].items() if k.startswith(prefix))
            for kind,prefix in (('lora','backbone.'),('head','decision_head.'))};prior=bundle['tensors']
        a=read(run/f'activation-{step:04d}.json')
        if a['error'] is not None or a['policy_id']!=current:raise ValueError('updated policy not activated')
        activation(a['receipt'],current)
        times=[]
        for rank in range(2):
            r=read(run/f'rank-{rank}/update-{step:04d}.json');parity=r['behavior_parity_history']
            if (r['update']!=step or r['real_queries']!=real_counts[index] or r['next_policy_id']!=current
                    or r['rollout']!=receipt or r['checkpoint'] is not save_now or len(parity)!=layout['accumulation']
                    or any(len(g)!=2 or any(p!=dict(passed=True,error=None) for p in g) for g in parity)
                    or not all(positive(r[k]) for k in ('total_seconds','actor_seconds','preparation_seconds'))):
                raise ValueError('complete DP2 parity/update/timing required')
            if states is not None:
                scope=validate_trainable_scope([(n,torch.nn.Parameter(v)) for n,v in states[rank]['parameters'].items()])
                if read(run/f'rank-{rank}/checkpoint-scope.json')!=scope:raise ValueError('trainable-only checkpoint scope differs')
            times.append(r['total_seconds'])
        summary.update(slowest_rank_step_seconds=max(times),changed_execution_tensors=changes,master_export=master)
        summaries.append(summary)
    if {p.name for p in journal.glob('batch-*')}!={f'batch-{i:08d}' for i in range(sequence)}:raise ValueError('unaccounted macro batch')
    if any(sum(s['changed_execution_tensors'][k] for s in summaries)==0 for k in ('lora','head')):raise ValueError('LoRA and head updates must both be exercised')
    for rank in range(2):
        r=read(run/f'rank-{rank}/result.json')
        if (r['passed'] is not True or r['start_update']!=start or r['stop_update']!=stop
                or len(r['executions'])!=113 or set(r['executions'].values())!={'bfloat16'}):raise ValueError('all-rank BF16 completion required')
    if start:
        from tools.reports.compare_context_dual_throughput_pair import equal
        original=Path(binding['parent_operations']['operations_root'])/'collected/run'
        for name in ('groups-0001.pt','execution-0002.pt',*[f'rank-{r}/checkpoint-0002.pt' for r in range(2)]):
            if not equal(torch.load(original/name,map_location='cpu',weights_only=True),
                         torch.load(run/name,map_location='cpu',weights_only=True)):
                raise ValueError('fresh-process replay tensor or optimizer differs: '+name)
        if (read(original/'batch-0001.json')!=read(run/'batch-0001.json')
                or read(original/'checkpoint-0002-committed.json')['joint_policy']['policy_id']!=current):
            raise ValueError('replayed schedule inputs or resulting policy differs')
    return dict(status='formal_dp2_outputs_audited' if formal else 'finite_dp2_efficiency_audited' if efficiency else 'finite_dp2_replay_audited' if start else 'finite_dp2_training_audited',updates=summaries,macros=macro_summaries,
        fresh_process_replay_verified=bool(start),training_dispatch_allowed=False,
        scope=('Full634 DP2 output audit; every policy/activation checked, master/Adam only at declared checkpoints; runtime transport and efficacy remain separate' if formal else
               'Finite DP2 journal/export/activation audit and explicit replay result; no efficacy or sustained speed claim'))


def audit_collected(c, *, validator=None, command_builder=None, output_auditor=None):
    """Re-audit immutable collected bytes without recopying a remote archive."""
    from tools.reports.context_macro_probe_operations import inspect_runtime
    validator=validator or validate;command_builder=command_builder or command
    output_auditor=output_auditor or audit_outputs
    manifest=validator(c);root=Path(c['operations_root']);collection=root/'collected'
    transport=read(root/'transport.json');inspections=read(root/'collected-inspect.json')
    if read(root/'terminal.json')['exit_code']!=0:raise ValueError('successful driver required')
    if file_hash(root/'results.tar')!=transport['archive_sha256']:raise ValueError('archive changed')
    if any(p.is_symlink() for p in collection.rglob('*')):raise ValueError('collection symlink forbidden')
    files={str(p.relative_to(collection)):file_hash(p) for p in collection.rglob('*') if p.is_file()}
    if files!=transport['files']:raise ValueError('complete collected bytes changed')
    for role in ('preflight','client','service-0','service-1'):
        h=read(root/(role+'-launch.json'))
        if h['argv']!=command_builder(c,role):raise ValueError('launch differs')
        inspect_runtime(inspections[role],h,terminal=True)
    package=Path(c['prepared'])/'package'
    if (package/'REPLAY.json').exists():
        parent=read(package/'REPLAY.json')['parent_container']
        if inspections['client']['Id']==parent:raise ValueError('fresh actor container required for replay')
    result=output_auditor(package,collection)
    result.update(files_sha256=manifest['files_sha256'],transport_sha256=file_hash(root/'transport.json'),
        containers={r:obs['Id'] for r,obs in inspections.items()},image=manifest['image'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['collect','audit'],default='collect')
    a=p.parse_args();c,_=load_config(a.config,a.local);torch.set_num_threads(1)
    if a.action=='collect':
        result=collect_and_audit(c,validator=validate,command_builder=command,
            output_auditor=audit_outputs,result_dirs=('run','journal','service-0','service-1'))
    else:
        from tools.reports.launch_context_batch_probe_candidate import write_new
        result=audit_collected(c);write_new(Path(c['operations_root'])/'audit.json',result)
    print(json.dumps(result,indent=2))
