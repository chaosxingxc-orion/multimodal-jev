"""Full-population fixed-policy evaluation over the training macro transport.

Caller authenticates the dataset, candidate definitions, processor and runtime
package. This layer checks actual tensor bindings and complete atomic batches;
it never loads labels, samples actions for learning, switches policies or scores
targets. Independent collection/runtime audits remain required.
"""
import copy
from itertools import islice
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.context_macro_rollout import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.span_request_identity import tensor_identity


def evaluate(pool, population, encode, *, candidate_ids, policy_id,
             runtime_identity, audio_token_id, macro_queries, output):
    """encode(label-free asset) returns tensors and an independently bound receipt.

    All queries use the same ordered candidate panel. Variable candidate panels
    (e.g. SFC functions) require a separate admitted task/input adapter.
    """
    candidates=list(candidate_ids)
    if (not candidates or any(type(x) is not str or not x for x in candidates)
            or len(set(candidates))!=len(candidates) or not population
            or type(macro_queries) is not int or not 1<=macro_queries<=pool.max_queries):
        raise ValueError('complete candidates/population and bounded macro required')
    ids=[]
    for asset in population:
        if (set(asset)!={'id','audio_sha256','request_sha256'}
                or any(type(v) is not str or not v for v in asset.values())):
            raise ValueError('only label-free immutable asset identities accepted')
        ids.append(asset['id'])
    if len(set(ids))!=len(ids):raise ValueError('unique complete population required')
    def check_policy():
        if pool.bundle['policy_id']!=policy_id or pool.runtime_identity!=runtime_identity:
            raise ValueError('fixed policy and declared runtime required')
    check_policy()
    root=Path(output);check_paths({'evaluation':root});root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();records=[];receipts=[];batches=[]
    try:
        commit_json(root/'population.json',dict(assets=population,candidate_ids=candidates,
            policy_id=policy_id,runtime_identity=runtime_identity))
        iterator=iter(population)
        while assets:=list(islice(iterator,macro_queries)):
            check_policy();rows=[];bound=[]
            for asset in assets:
                row,receipt=encode(copy.deepcopy(asset))
                if set(row)!={'inputs','positions'}:raise ValueError('label-free tensor row required')
                if row['positions'].ndim!=2 or tuple(row['positions'].shape)!=(1,len(candidates)):
                    raise ValueError('complete candidate positions required')
                identity=dict(policy_id=policy_id,positions=row['positions'][0].tolist(),
                    inputs={k:tensor_identity(v) for k,v in sorted(row['inputs'].items())})
                if (any(receipt.get(k)!=v for k,v in asset.items())
                        or receipt.get('candidate_ids')!=candidates
                        or receipt.get('inputs')!=identity['inputs']
                        or receipt.get('positions')!=identity['positions']):
                    raise ValueError('encoded inputs differ from candidate/asset receipt')
                rows.append(row);bound.append((identity,copy.deepcopy(receipt)))
            batch_id=f'evaluation-{len(batches):08d}'
            values=pool.score_macro(batch_id,rows,audio_token_id=audio_token_id)
            check_policy();macro=copy.deepcopy(pool.last_receipt)
            if (len(values)!=len(assets) or macro['batch_id']!=batch_id
                    or macro['status']!='complete_macro_inference'
                    or macro['policy_id']!=policy_id or macro['actor_update_performed'] is not False
                    or macro['completed_indices']!=list(range(len(assets)))
                    or macro['result_sha256']!=digest(values)):
                raise ValueError('complete unchanged macro result required')
            predictions=[];batch_receipts=[]
            for asset,row,value,(identity,receipt) in zip(assets,rows,values,bound,strict=True):
                validate_logits_record(value)
                after=dict(policy_id=policy_id,positions=row['positions'][0].tolist(),
                    inputs={k:tensor_identity(v) for k,v in sorted(row['inputs'].items())})
                if (after!=identity or value['request_identity']!=identity
                        or value['request_id']!=digest(identity) or value['policy_id']!=policy_id
                        or len(value['prob'])!=len(candidates)):
                    raise ValueError('prediction/input/policy binding differs')
                predictions.append(dict(value,id=asset['id']))
                batch_receipts.append(dict(receipt,engine_request_id=value['request_id']))
            # No partial batch becomes a published prediction file.
            name=f'batch-{len(batches):08d}.json'
            commit_json(root/name,dict(predictions=predictions,input_receipts=batch_receipts,macro_receipt=macro))
            records.extend(predictions);receipts.extend(batch_receipts)
            batches.append(dict(path=name,sha256=file_hash(root/name),queries=len(assets)))
            atomic_json(root/'progress.json',dict(completed=len(records),total=len(population),
                seconds=time.monotonic()-started,policy_id=policy_id))
        if [r['id'] for r in records]!=ids:raise ValueError('complete ordered population required')
        check_policy()
        commit_json(root/'predictions.json',dict(records=records))
        commit_json(root/'input-receipts.json',receipts)
        result=dict(policy_id=policy_id,runtime_identity=runtime_identity,
            recordings=len(records),decisions=len(candidates),batches=batches,
            population_sha256=file_hash(root/'population.json'),
            predictions_sha256=file_hash(root/'predictions.json'),
            input_receipts_sha256=file_hash(root/'input-receipts.json'),
            seconds=time.monotonic()-started,
            scope='Complete fixed-policy macro inference; independent dataset/runtime audit and scoring required')
        commit_json(root/'result.json',result)
        commit_json(root/'terminal.json',dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc)))
        raise
