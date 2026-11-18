"""Build immutable TRAIN reference results from label-free source assets.

The caller admits the actual producer runtime. This builder authenticates
coverage, request binding and immutable policy; it never starts an engine.
"""
import time
import copy

from .context_candidate_task import compile_candidates
from .context_logits_wire import validate_logits_record
from .context_reference_dataset import digest,publish,validate_identity,validate_record
from .span_request_identity import tensor_identity


def build(root, *, assets, candidates, requirements, preparer, engine, identity,
          macro_queries, audio_token_id, producer):
    """assets is {sample_id: {audio, audio_sha256}} with no label access."""
    identity=copy.deepcopy(identity)
    if (not isinstance(assets,dict) or not assets
            or any(type(k)is not str or not k or set(v)!={'audio','audio_sha256'} for k,v in assets.items())
            or type(macro_queries)is not int or not 1<=macro_queries<=4096
            or type(audio_token_id)is not int or audio_token_id<0):
        raise ValueError('complete label-free asset population and bounded macro required')
    if identity.get('candidate_ids')!=[c['id'] for c in candidates]:
        raise ValueError('candidate order differs from reference contract')
    if 'population_sha256' in identity:
        raise ValueError('population digest is derived from actual prepared inputs')
    validate_identity(dict(identity,population_sha256='0'*64))
    policy=identity['policy_id']
    if engine.bundle['policy_id']!=policy or engine.runtime_identity!=identity['runtime_identity']:
        raise ValueError('fixed producer policy/runtime differs')
    population={};records={};pending={};batches=[]
    started=time.monotonic();preparation_seconds=0.;scoring_seconds=0.

    def flush():
        nonlocal scoring_seconds
        if not pending:return
        if engine.bundle['policy_id']!=policy or engine.runtime_identity!=identity['runtime_identity']:
            raise ValueError('reference producer changed before inference')
        keys=list(pending);inputs=[pending[k][0] for k in keys];begin=time.monotonic()
        outputs=engine.score_many(inputs,audio_token_id=audio_token_id)
        elapsed=time.monotonic()-begin;scoring_seconds+=elapsed
        if len(outputs)!=len(keys) or engine.bundle['policy_id']!=policy:
            raise ValueError('incomplete or switched reference macro')
        for key,output in zip(keys,outputs,strict=True):
            validate_logits_record(output)
            record={k:output[k] for k in ('request_id','request_identity','prob','policy_id')}
            validate_record(record,dict(identity,population_sha256='0'*64))
            if record['request_id']!=key or record['request_identity']!=pending[key][1]:
                raise ValueError('reference result input/order differs')
            records[key]=record
        batches.append(dict(index=len(batches),queries=len(keys),seconds=elapsed))
        pending.clear()

    for sample_id in sorted(assets):
        asset=assets[sample_id];begin=time.monotonic()
        request=compile_candidates(asset['audio'],candidates,requirements,semantics='exclusive_event')
        value=preparer((request,asset['audio_sha256']))
        if not value.cache_hit:
            raise ValueError('reference build requires offline-prepared input cache')
        from .encoded_audio_cache import canonical
        if value.request!=canonical(request.model_input()):
            raise ValueError('prepared reference request differs from source')
        binding=dict(policy_id=policy,positions=value.positions[0].tolist(),
                     inputs={k:tensor_identity(v) for k,v in sorted(value.inputs.items())})
        key=digest(binding);population[sample_id]=key
        if key not in records and key not in pending:
            pending[key]=(dict(inputs=value.inputs,positions=value.positions),binding)
        preparation_seconds+=time.monotonic()-begin
        if len(pending)==macro_queries:flush()
    flush()
    if engine.runtime_identity!=identity['runtime_identity']:
        raise ValueError('reference runtime changed during build')
    final_identity=dict(identity,population_sha256=digest(population))
    timing=dict(preparation_seconds=preparation_seconds,scoring_seconds=scoring_seconds,
                build_before_publication_seconds=time.monotonic()-started)
    publication=publish(root,identity=final_identity,records=list(records.values()),population=population,
        producer=dict(producer,timing=timing,macro_batches=batches,actor_updates=0))
    return dict(publication,identity=final_identity,population=population,timing=timing,
                batches=len(batches),numerical_admission_claim=False)
