"""Fixed capacity probe contract; no test labels or prompt selection."""
import hashlib
import json
from pathlib import Path

from .baseline import file_hash

PROCESSOR_RESULT_SHA='e1bc9e578db1b14d911978607dc61c5cdadf8a12630fa103e18281713ce54ec5'
ASSET_AUDIT_SHA='3c08bfd664f6b445054778c281e219eeb362f7b051059f59cc988453c5530930'
VERSIONS={'torch':'2.10.0+cpu','torch-npu':'2.10.0.post7','transformers':'5.14.1',
          'vllm':'0.30.0+empty','vllm-ascend':'0.30.0rc2'}
ENGINE=dict(dtype='bfloat16',tensor_parallel_size=2,seed=17,max_model_len=32768,
    max_num_seqs=1,max_num_batched_tokens=32768,gpu_memory_utilization=.85,
    enforce_eager=True,trust_remote_code=False,limit_mm_per_prompt={'audio':7,'image':0,'video':0},
    enable_prefix_caching=False)


def prior_records(root):
    root=Path(root)
    if file_hash(root/'result.json')!=PROCESSOR_RESULT_SHA:
        raise ValueError('registered successful actual processor result required')
    result=json.loads((root/'result.json').read_text())
    if (result['status']!='passed' or len(result['records'])!=4
            or result['identity_sha256']!=file_hash(root/'identity.json')
            or json.loads((root/'terminal.json').read_text())['exit_code']!=0):
        raise ValueError('complete successful processor evidence required')
    identity=json.loads((root/'identity.json').read_text())
    records={r['selection']:r for r in result['records']}
    if len(records)!=4 or {k:r['id'] for k,r in records.items()}!=identity['selected']:
        raise ValueError('complete boundary selection required')
    return records,identity


def validate_request(request,receipts,prior):
    if (hashlib.sha256(request['prompt'].encode()).hexdigest()!=prior['prompt_sha256']
            or receipts!=prior['audio_receipts']
            or len(request['multi_modal_data']['audio'])!=len(receipts)
            or any(rate!=16000 for wave,rate in request['multi_modal_data']['audio'])):
        raise ValueError('engine input differs from accepted processor request')


def accept_response(responses,prior):
    """One-token prefill probe, not task inference or 1024-token decode proof."""
    if len(responses)!=1 or not responses[0].finished or len(responses[0].outputs)!=1:
        raise ValueError('one terminal engine response required')
    response=responses[0];answer=response.outputs[0]
    if (len(response.prompt_token_ids)!=prior['input_tokens']
            or hashlib.sha256(json.dumps(list(response.prompt_token_ids)).encode()).hexdigest()!=prior['input_ids_sha256']):
        raise ValueError('engine expanded token IDs differ from accepted processors')
    if (answer.finish_reason not in ('stop','length') or len(answer.token_ids)!=1
            or any(type(t) is not int or t<0 for t in answer.token_ids)):
        raise ValueError('exactly one generated token required for capacity probe')
    return dict(token_ids=list(answer.token_ids),finish_reason=answer.finish_reason,
                input_tokens=len(response.prompt_token_ids),input_ids_sha256=prior['input_ids_sha256'])
