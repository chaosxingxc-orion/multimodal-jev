"""Reconstruct complete evaluation outputs from rollout journals and input evidence.

Expected population/receipts must come from the independently authenticated
dataset/processor audit, not from this inference run's own receipts. Transport,
container admission and HF/vLLM parity are separate checks.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.context_macro_rollout import digest
from multimodal_jev.span_request_identity import tensor_identity
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths


def audit(output,journal,*,population,candidate_ids,input_receipts,policy_id,runtime_identity,macro_queries,replicas=2):
    output=Path(output);journal=Path(journal)
    def read(root,name):
        path=root/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError('contained regular evidence file required')
        return json.loads(path.read_text())
    if (not population or len({r['id'] for r in population})!=len(population)
            or len(input_receipts)!=len(population) or not candidate_ids
            or len(set(candidate_ids))!=len(candidate_ids)
            or type(replicas) is not int or not 1<=replicas<=8
            or type(macro_queries) is not int or macro_queries<1):
        raise ValueError('independently bound complete population and candidate panel required')
    if read(output,'terminal.json')!={'exit_code':0}:
        raise ValueError('successful full evaluation required')
    expected_population=dict(assets=population,candidate_ids=candidate_ids,
        policy_id=policy_id,runtime_identity=runtime_identity)
    if read(output,'population.json')!=expected_population:
        raise ValueError('population/policy/runtime differs from independent expectation')
    result=read(output,'result.json')
    if (result['policy_id']!=policy_id or result['runtime_identity']!=runtime_identity
            or result['recordings']!=len(population) or result['decisions']!=len(candidate_ids)
            or not math.isfinite(result['seconds']) or result['seconds']<=0):
        raise ValueError('result identity/count/timer differs')
    for name,key in [('population.json','population_sha256'),('predictions.json','predictions_sha256'),
                     ('input-receipts.json','input_receipts_sha256')]:
        if file_hash(output/name)!=result[key]:raise ValueError('evaluation artifact hash differs')
    expected_batches=(len(population)+macro_queries-1)//macro_queries
    if len(result['batches'])!=expected_batches:raise ValueError('all planned batches required')
    records=[];receipts=[];batch_ids=[]
    for index,entry in enumerate(result['batches']):
        name=f'batch-{index:08d}.json';batch_id=f'evaluation-{index:08d}';batch_ids.append(batch_id)
        start=index*macro_queries;assets=population[start:start+macro_queries]
        independent=input_receipts[start:start+macro_queries];n=len(assets)
        if entry!={'path':name,'sha256':file_hash(output/name),'queries':n}:
            raise ValueError('batch order/hash/size differs')
        saved=read(output,name);root=journal/batch_id
        if root.is_symlink() or (root/'FAILED.json').exists():raise ValueError('failed/aliased batch forbidden')
        manifest=read(root,'INPUT.json');complete=read(root,'COMPLETE.json')
        if (manifest['batch_id']!=batch_id or manifest['policy_id']!=policy_id
                or manifest['runtime_identity']!=runtime_identity
                or manifest['expected_indices']!=list(range(n)) or len(manifest['inputs'])!=n):
            raise ValueError('macro input identity or population differs')
        for asset,receipt,identity in zip(assets,independent,manifest['inputs'],strict=True):
            if (any(receipt.get(k)!=v for k,v in asset.items())
                    or receipt['candidate_ids']!=candidate_ids or len(receipt['positions'])!=len(candidate_ids)):
                raise ValueError('independent receipt belongs to different asset/menu')
            expected=dict(receipt['inputs'],positions=tensor_identity(torch.tensor([receipt['positions']],dtype=torch.long)))
            if identity!=expected:raise ValueError('actual macro tensor identity differs from input audit')
        ordered=[None]*n;hashes={};assigned=[]
        for j,job in enumerate(manifest['jobs']):
            rollout_id=f'rollout-{j:04d}'
            if (job['rollout_id']!=rollout_id or type(job['replica_id']) is not int or not 0<=job['replica_id']<replicas
                    or not job['indices'] or any(type(i) is not int or not 0<=i<n for i in job['indices'])
                    or type(job['estimated_cost']) is not int or job['estimated_cost']<=0):
                raise ValueError('bounded whole-rollout ownership required')
            recorded=read(root,rollout_id+'.json')
            if (any(recorded[k]!=v for k,v in job.items()) or recorded['policy_id']!=policy_id
                    or len(recorded['records'])!=len(job['indices'])
                    or not math.isfinite(recorded['seconds']) or recorded['seconds']<=0):
                raise ValueError('rollout ownership/result/timer differs')
            hashes[rollout_id]=digest(recorded)
            for slot,row in zip(job['indices'],recorded['records'],strict=True):
                if ordered[slot] is not None:raise ValueError('duplicate rollout slot')
                receipt=independent[slot]
                identity=dict(policy_id=policy_id,positions=receipt['positions'],inputs=receipt['inputs'])
                validate_logits_record(row)
                if (row['policy_id']!=policy_id or row['request_identity']!=identity
                        or row['request_id']!=digest(identity) or len(row['prob'])!=len(candidate_ids)):
                    raise ValueError('actual output tensor/policy binding differs')
                ordered[slot]=row;assigned.append(slot)
        if sorted(assigned)!=list(range(n)):raise ValueError('complete unique rollout coverage required')
        if (saved['macro_receipt']!=complete or complete['batch_id']!=batch_id
                or complete['policy_id']!=policy_id or complete['status']!='complete_macro_inference'
                or complete['actor_update_performed'] is not False or complete['completed_indices']!=list(range(n))
                or complete['input_sha256']!=digest(manifest) or complete['rollout_hashes']!=hashes
                or complete['result_sha256']!=digest(ordered)
                or not math.isfinite(complete['seconds']) or complete['seconds']<=0):
            raise ValueError('complete receipt differs from independently reconstructed rollouts')
        predicted=[dict(row,id=asset['id']) for row,asset in zip(ordered,assets,strict=True)]
        expected_receipts=[dict(r,engine_request_id=row['request_id']) for r,row in zip(independent,ordered,strict=True)]
        if saved['predictions']!=predicted or saved['input_receipts']!=expected_receipts:
            raise ValueError('saved predictions or input receipts differ')
        records.extend(predicted);receipts.extend(expected_receipts)
    if sorted(p.name for p in journal.glob('evaluation-*'))!=batch_ids:
        raise ValueError('unexpected or missing evaluation journal batch')
    if sorted(p.name for p in output.glob('batch-*.json'))!=[b['path'] for b in result['batches']]:
        raise ValueError('unexpected or missing saved batch')
    if read(output,'predictions.json')!={'records':records} or read(output,'input-receipts.json')!=receipts:
        raise ValueError('aggregate differs from complete independently bound batches')
    return dict(status='complete_macro_evaluation_outputs_audited',policy_id=policy_id,
        recordings=len(records),decisions=len(candidate_ids),batches=expected_batches,
        predictions_sha256=result['predictions_sha256'],input_receipts_sha256=result['input_receipts_sha256'],
        scope='Offline output/journal/input evidence binding; caller authenticates dataset, processor, transport, runtime and parity. No task score or transfer claim.')


def run(config,provenance):
    """Explicit independent input evidence; never discover it from predictions."""
    expected={}
    for key in ('population','candidate_ids','input_receipts'):
        entry=config[key]
        if set(entry)!={'path','sha256'} or file_hash(entry['path'])!=entry['sha256']:
            raise ValueError('independent input artifact identity differs: '+key)
        expected[key]=json.loads(Path(entry['path']).read_text())
    report=audit(config['inference'],config['journal'],**expected,
        policy_id=config['policy_id'],runtime_identity=config['runtime_identity'],
        macro_queries=config['macro_queries'])
    target=Path(config['output']);check_paths({'audit':target})
    target.parent.mkdir(parents=True,exist_ok=True)
    report.update(provenance=provenance,independent_inputs={k:config[k] for k in expected})
    commit_json(target,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    run(config,provenance)
