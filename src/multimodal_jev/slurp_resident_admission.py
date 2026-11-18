"""Read-only recovered FSDP check against accepted HF full-candidate outputs."""
import json
from pathlib import Path
from .baseline import file_hash,atomic_json


def check_restored_policy(model, processor, config, output, *, rank, device):
    import torch
    import torch.distributed as dist
    from .span_contract import SpanRequest,SpanQuestion,encode_span_request,span_labels
    from .task_schema import DecisionSchema
    from .audio_policy import load_audio
    from .span_request_identity import tensor_identity
    from .span_parity import compare_span_policies
    gate=Path(config['previous_gate'])
    gate_config=json.loads((gate/'config.json').read_text())
    artifact_path=Path(gate_config['inputs']);hf_path=gate/'run/hf/result.json'
    for path in (gate/'config.json',artifact_path,hf_path):
        if config['pins'].get(str(path))!=file_hash(path):
            raise ValueError('pinned independently accepted inputs and HF probabilities required')
    artifact=json.loads(artifact_path.read_text());hf=json.loads(hf_path.read_text())
    if (not hf['passed'] or hf['base_id']!=config['base_id']
            or hf['adapter_id']!=gate_config['adapter_id'] or artifact['requests']!=256):
        raise ValueError('complete accepted predecessor policy required')
    references={r['id']:r for r in hf['records']}
    rows=artifact['records']
    if len(rows)!=256 or len(references)!=256 or {r['id'] for r in rows}!=references.keys():
        raise ValueError('exact full-prefix admission population required')
    records=[]
    # Exactly 64 forwards per rank, no backward or mutation, same collective order.
    with torch.no_grad():
        for row in rows[rank::4]:
            raw=row['request']
            request=SpanRequest(raw['audio'],tuple(SpanQuestion(q['id'],DecisionSchema.from_dict(q['schema']))
                for q in raw['questions']),raw['final_id'],tuple(raw['requirements']))
            if file_hash(request.audio)!=row['audio_sha256']:raise ValueError('admission waveform changed')
            audio,receipt=load_audio(request.audio,return_receipt=True)
            inputs,positions=encode_span_request(processor,request,audio)
            mapping=span_labels(processor.tokenizer,request.questions)
            if (receipt!=row['audio_processing'] or positions.tolist()!=row['positions']
                    or json.loads(json.dumps(mapping))!=row['candidate_mapping']
                    or any(tensor_identity(inputs[k])!=row[k] for k in ('input_ids','input_features','input_features_mask'))):
                raise ValueError('recovered actor input differs from accepted HF input')
            inputs={k:v.to(device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
            probs=model(inputs,positions.to(device),[m['token_ids'] for m in mapping])[1]
            logp=[p[0].cpu().tolist() for p in probs]
            parity=compare_span_policies(references[row['id']]['candidate_logprobs'],logp)
            records.append(dict(id=row['id'],candidate_logprobs=logp,parity=parity))
    local=dict(rank=rank,requests=len(records),passed=len(records)==64 and all(r['parity']['passed'] for r in records),
        maximum_logprob_error=max(r['parity']['maximum_logprob_error'] for r in records))
    peers=[None]*4;dist.all_gather_object(peers,local)
    atomic_json(output/'restored-policy.json',dict(**local,records=records,ranks=peers,
        scope='Recovered resident FSDP vs accepted HF over exact original prefix inputs before any new optimizer update.'))
    if any(not p['passed'] for p in peers):
        raise ValueError('recovered resident FSDP policy failed accepted HF parity')
    return local
