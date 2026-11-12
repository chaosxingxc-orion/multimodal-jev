"""Bind admitted intent supervision to exact accepted audio Span policies."""
import hashlib
import json
from pathlib import Path

from .baseline import file_hash
from .slurp_intent_pilot import load_pilot


def schedule(items,seed,step,global_requests=8):
    """Deterministic cyclic shuffle, without dropping the 89th source.

    Each fixed ordering cycles; every consecutive batch has distinct sources.
    Class balance comes from outer request weights, never extra resampling.
    """
    if (type(seed)!=int or type(step)!=int or step<0 or type(global_requests)!=int
            or not 1<=global_requests<=len(items)):
        raise ValueError('valid fixed seed/step and bounded complete batch required')
    if len({x['id'] for x in items})!=len(items):raise ValueError('distinct source requests required')
    ordered=sorted(items,key=lambda x:hashlib.sha256(f'{seed}:{x["id"]}'.encode()).hexdigest())
    return [ordered[(step*global_requests+i)%len(ordered)] for i in range(global_requests)]


def prepare(config,processor,policy_id,step):
    """Compile all global rows on controller before splitting across DP ranks.

    Targets/weights never enter encode_span_request or bind_span_request.
    Full per-request HF/vLLM evidence must already exist for this policy.
    """
    import torch
    from .audio_policy import load_audio
    from .span_contract import encode_span_request,span_labels
    from .span_request_identity import bind_span_request
    from .span_parity import compare_span_policies
    from .policy_identity import checkpoint_identity
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('actor input pin changed')
    if 'resident_behavior' in config:
        from .intent_resident_behavior import authenticate
        authenticate(config,policy_id)
    root=Path(config['acceptance_root'])
    # Never accept a path merely because an old READY exists.
    required=[root/'config.json',root/'run/READY.json',root/'run/terminal.json',
        root/'run/hf/result.json',root/'run/hf/terminal.json',root/'run/vllm/result.json',root/'run/vllm/terminal.json']
    if any(config['pins'].get(str(p))!=file_hash(p) for p in required):raise ValueError('pinned complete independent gate required')
    read=lambda p:json.loads(p.read_text())
    for name in ('terminal.json','hf/terminal.json','vllm/terminal.json'):
        if read(root/'run'/name)['exit_code']!=0:raise ValueError('gate did not finish')
    gate_config=read(root/'config.json')
    if gate_config.get('runtime_profile')!=config['runtime_profile']:
        raise ValueError('runtime profile differs')
    ready=read(root/'run/READY.json');hf=read(root/'run/hf/result.json');vl=read(root/'run/vllm/result.json')
    expected_policy=checkpoint_identity(config['model_path'])[0]+':'+checkpoint_identity(config['initial_adapter'])[0]
    if policy_id!=expected_policy or not ready['passed'] or ready['result_sha256']!=file_hash(root/'run/vllm/result.json'):
        raise ValueError('policy acceptance differs')
    if ready['base_id']+':'+ready['adapter_id']!=policy_id:
        raise ValueError('READY policy differs')
    for data in (hf,vl):
        if not data['passed'] or data['base_id']+':'+data['adapter_id']!=policy_id:raise ValueError('HF/vLLM policy differs')
    references={r['id']:r for r in hf['records']};behaviors={r['id']:r for r in vl['records']}
    if (len(references)!=ready['requests'] or len(references)!=len(hf['records'])
            or len(behaviors)!=len(vl['records']) or set(references)!=set(behaviors)):
        raise ValueError('complete distinct matching gate population required')
    items,admission=load_pilot(config['pilot'],config['manifest_sha256'],config['arm'])
    # P shares Q's exact model input, but uses its own external reward weights.
    wire_arm='D' if config['arm']=='D' else 'Q'
    selected=schedule(items,config['seed'],step,config['global_requests'])
    prepared=[]
    for item in selected:
        request=item['request'];key=item['id']+'::'+wire_arm
        if key not in references:raise ValueError('selected input was not accepted')
        audio,receipt=load_audio(request.audio,return_receipt=True)
        inputs,positions=encode_span_request(processor,request,audio)
        candidates=[m['token_ids'] for m in span_labels(processor.tokenizer,request.questions)]
        rid,identity=bind_span_request(request,receipt,inputs,positions,candidates,policy_id)
        ref=references[key];behavior=behaviors[key]
        if any(r['request_id']!=rid or r['request_identity']!=identity for r in (ref,behavior)):
            raise ValueError('exact accepted actor request differs')
        audit=compare_span_policies(ref['candidate_logprobs'],behavior['candidate_logprobs'])
        if not audit['passed']:raise ValueError('accepted candidate probabilities differ')
        target=item['target']
        prepared.append(dict(request_id=rid,policy_id=policy_id,inputs=inputs,positions=positions,candidates=candidates,
            behavior=tuple(torch.tensor([p],dtype=torch.float32) for p in behavior['candidate_logprobs']),
            targets=torch.tensor([target['targets']],dtype=torch.long),supervised=torch.tensor([target['supervised']],dtype=torch.bool),
            span_weights=torch.tensor([target['span_weights']],dtype=torch.float32),request_weight=target['request_weight']))
    return prepared,dict(admission=admission,step=step,seed=config['seed'],recording_ids=[x['id'] for x in selected],
        source_groups=[x['target']['source_group'] for x in selected],
        schedule='fixed hash ordering, cyclic full batches, no dropped tail or extra balanced resampling')
