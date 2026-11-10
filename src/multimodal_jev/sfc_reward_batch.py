"""Complete SFC learning replicate rewards and same-input GRPO identities.

This prepares scalar rewards and grouping metadata, not tensors or advantages.
The caller must obtain prompt evidence from the actual processor/engine and
bind files/checkpoints in its run protocol before calling this pure adapter.
"""
import hashlib
import math

from .consumption import canonical
from .sfc_teacher_feedback import validate_partition
from .sfc_jev_reward import bundle_rewards


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def valid_hash(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def compose(all_requests,partition,requests,targets,records,prompt_evidence,*,policy_sha256,
            instruction_sha256,group_size,tokenizer,mode='independent',bonus_weight=0.2):
    """Return rewards/uids in original record order, independent of arrival order.

    prompt_evidence is a pre-rollout dictionary keyed by (bundle, decision).
    Each value binds the rendered prompt and expanded input IDs. All K responses
    for every decision and all original turns of selected learning groups are
    required. Replicate numbers are fixed before sampling, never sorted by reward.
    """
    if type(group_size) is not int or group_size<2:
        raise ValueError('at least two registered sampling replicates required')
    if not valid_hash(policy_sha256) or not valid_hash(instruction_sha256):
        raise ValueError('pinned policy and instruction hashes required')
    population=validate_partition(all_requests,partition)
    by_id={r['id']:r for r in requests};gold={t['id']:t for t in targets}
    learning=set(partition['learning']);groups={r['source_group'] for r in requests}
    if (not requests or len(by_id)!=len(requests) or len(gold)!=len(targets) or set(gold)!=set(by_id)
            or not groups<=learning or set(by_id)!={r['id'] for r in all_requests if r['source_group'] in groups}
            or any(r['split']!='train_id' or population.get(r['id'])!=r for r in requests)):
        raise ValueError('complete original learning conversations and aligned targets required')
    expected={(r['id'],d['id']) for r in requests for d in r['input']['decisions']}
    if set(prompt_evidence)!=expected:raise ValueError('exact pre-rollout prompt roster required')
    uids={};request_hashes={uid:digest(r) for uid,r in by_id.items()}
    for key,evidence in prompt_evidence.items():
        if (set(evidence)!={'prompt_sha256','input_ids_sha256'}
                or any(not valid_hash(v) for v in evidence.values())):
            raise ValueError('exact rendered and expanded prompt hashes required')
        uids[key]=digest(dict(request_sha256=request_hashes[key[0]],decision_id=key[1],
            policy_sha256=policy_sha256,instruction_sha256=instruction_sha256,**evidence))
    fields={'example_id','decision_id','request_sha256','replicate','policy_sha256','instruction_sha256',
            'prompt_sha256','input_ids_sha256','raw_text','token_ids','log_probs','finish_reason'}
    found={};aligned={};identities=[]
    for index,record in enumerate(records):
        if set(record)!=fields:raise ValueError('exact sampling record fields required')
        key=record['example_id'],record['decision_id'];rep=record['replicate']
        if key not in expected or type(rep) is not int or not 0<=rep<group_size or (*key,rep) in found:
            raise ValueError('unknown, duplicate or invalid sampling replicate')
        if (record['request_sha256']!=request_hashes[key[0]] or record['policy_sha256']!=policy_sha256
                or record['instruction_sha256']!=instruction_sha256
                or any(record[k]!=v for k,v in prompt_evidence[key].items())):
            raise ValueError('stale policy or changed audio/prompt identity')
        ids=record['token_ids'];logps=record['log_probs']
        if (type(ids) is not list or not ids or any(type(t) is not int or not 0<=t<len(tokenizer) for t in ids)
                or type(logps) is not list or len(logps)!=len(ids)
                or any(type(v) not in (int,float) or not math.isfinite(v) or v>1e-5 for v in logps)
                or record['finish_reason'] not in ('stop','length') or type(record['raw_text']) is not str):
            raise ValueError('complete finite token-aligned sampling output required')
        if tokenizer.decode(ids,skip_special_tokens=True)!=record['raw_text']:
            raise ValueError('reward text differs from sampled tokens')
        found[(*key,rep)]=index
        aligned.setdefault((key[0],rep),[]).append({k:record[k] for k in
            ('example_id','decision_id','request_sha256','raw_text')})
        identities.append(uids[key])
    if set(found)!={(*key,rep) for key in expected for rep in range(group_size)}:
        raise ValueError('every decision needs the complete registered replicate group')
    rewards=[None]*len(records);components=[None]*len(records);summaries=[]
    for request in requests:
        for rep in range(group_size):
            result=bundle_rewards(request,gold[request['id']],aligned[request['id'],rep],
                learning_groups=learning,mode=mode,bonus_weight=bonus_weight)
            for item in result['decisions']:
                index=found[request['id'],item['decision_id'],rep]
                rewards[index]=item['total'];components[index]=item
            summaries.append(dict(example_id=request['id'],replicate=rep,
                **{k:result[k] for k in ('correct_function_triples','native_calls_exact','all_decisions_correct')}))
    return dict(rewards=rewards,uid=identities,components=components,replicate_summaries=summaries,
        group_size=group_size,prompt_groups=len(expected),bundles=len(requests),source_groups=len(groups),
        response_tokens=sum(len(r['token_ids']) for r in records),mode=mode,bonus_weight=bonus_weight,
        scope='Reward composition and exact same-input grouping only. No advantage tensors, '
        'waveform processing, actor/vLLM probability parity, distributed execution or optimizer update.')
