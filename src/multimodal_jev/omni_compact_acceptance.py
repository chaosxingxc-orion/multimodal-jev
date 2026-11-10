"""Admit only the original complete six-row frozen compact-key candidate.

The caller captures actual Docker inspection and pins its hash before dispatch.
The saved inspection alone is not an independent origin of trust.
"""
import json
from pathlib import Path

from .baseline import file_hash

CONTAINER='742651b35fe69b6f7679a56136fd3643c03c07f915799937f70d6e6d87bd9da4'
IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
INTENT='07902e61933c77a245530c681110a9cf39842485a0c203159244caea301b42e9'
PRECISION_IDENTITY='e39bdebee8d2d434df8a51efc9d57b4f697462f0a67e8577d72f7f2e97cae9dd'


def require_candidate(root,inspection,inspection_hash):
    import torch
    root=Path(root);read=lambda p:json.loads(Path(p).read_text())
    if not inspection or not inspection_hash or file_hash(inspection)!=inspection_hash or file_hash(root/'intent.json')!=INTENT:
        raise ValueError('fixed candidate intent and coordinator inspection required')
    plan=read(root/'intent.json');launch=read(root/'launch.json');item=read(inspection)
    argv=plan['argv'];host=item['HostConfig'];state=item['State']
    if (launch['container_id']!=CONTAINER or launch['intent_sha256']!=INTENT
            or launch['source_commit']!=plan['source_commit'] or launch['image']!=IMAGE
            or item['Id']!=CONTAINER or item['Image']!=IMAGE or plan['image']!=IMAGE
            or state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']
            or host['NanoCpus']!=16*10**9 or host['Memory']!=160*1024**3
            or host['NetworkMode']!='none' or host['Devices'] or host['Privileged']
            or item['Config']['Cmd']!=argv[argv.index(IMAGE)+1:]
            or set(host['Binds'])!={argv[i+1] for i,v in enumerate(argv) if v=='-v'}
            or not all(argv[i+1] in item['Config']['Env'] for i,v in enumerate(argv) if v=='-e')):
        raise ValueError('exact successful original candidate container required')
    for name,pin in plan['source_files'].items():
        if file_hash(Path(plan['source_root'])/name)!=pin:raise ValueError('candidate source changed')
    run=root/'run';identity=read(run/'identity.json');result=read(run/'result.json')
    if (read(run/'terminal.json')!={'exit_code':0} or result['status']!='diagnostic_complete'
            or result['all_six_response_gates_passed'] is not True or result['threshold']!=.1 or result['weight_updates']!=0
            or identity['response_logp_max_error_threshold']!=.1 or identity['preflight_only'] is not False
            or identity['targets_read'] is not False or identity['weight_updates']!=0
            or len(identity['keys'])!=6 or len(result['reports'])!=6):
        raise ValueError('all six frozen candidate rows must pass unchanged threshold')
    for key,name in [('runner_sha256','tools/jev/diagnose_omni_compact_keys.py'),
                     ('adapter_sha256','src/multimodal_jev/omni_compact_attention.py')]:
        if identity[key]!=plan['source_files'][name]:raise ValueError('candidate executed source differs')
    precision=Path(argv[argv.index('--precision-root')+1])/'run'
    if file_hash(precision/'identity.json')!=PRECISION_IDENTITY or identity['precision_identity_sha256']!=PRECISION_IDENTITY:
        raise ValueError('original complete precision reference required')
    original=read(precision/'identity.json')
    if (identity['keys']!=original['keys'] or identity['response_ids']!=original['response_ids']
            or identity['model_files']!=original['model_files']
            or identity['padding_counts']!=[154,252,0,251,519,402]):raise ValueError('candidate population or model differs')
    labels=torch.tensor(identity['response_ids']).unsqueeze(-1);errors=[]
    for row,report in enumerate(result['reports']):
        if (report!=read(run/f'row-{row:02d}.json') or report['row']!=row or report['key']!=identity['keys'][row]
                or report['padding']!=identity['padding_counts'][row] or report['response_gate_passed'] is not True):
            raise ValueError('candidate complete ordered reports required')
        width=original['prompt_width']+len(labels);valid=width-report['padding']
        expected=[]
        for layer in range(48):
            e=dict(layer=layer,mode='unchanged_no_mask' if valid==width else 'compact_masked_keys',
                   query_length=width,key_lengths=[valid])
            if valid!=width:e['original_key_length']=width
            expected.append(e)
        if report['valid_tokens']!=valid or report['attention_records']!=expected:
            raise ValueError('all 48 compact-key text layers required')
        p=run/f'row-{row:02d}-logits.pt'
        if file_hash(p)!=report['logits_sha256']:raise ValueError('candidate logits changed')
        candidate=torch.load(p,map_location='cpu',weights_only=True)
        if candidate.ndim!=2 or candidate.shape[0]!=len(labels) or candidate.dtype!=torch.bfloat16 or not torch.isfinite(candidate).all():
            raise ValueError('finite complete BF16 candidate logits required')
        logp=candidate.float().log_softmax(-1).gather(-1,labels).squeeze(-1)
        if not torch.allclose(logp,torch.tensor(report['fp32_log_probs']),rtol=0,atol=1e-7):
            raise ValueError('candidate probabilities differ from saved logits')
        for cell in ('audio_padding','text_padding'):
            path=precision/f'row-{row:02d}-{cell}-logits.pt';pin=read(precision/f'row-{row:02d}-{cell}.json')['logits_sha256']
            c=report['comparisons'][cell]
            if file_hash(path)!=pin or pin!=c['reference_sha256']:raise ValueError('original logits changed')
            ref=torch.load(path,map_location='cpu',weights_only=True)
            if ref.shape!=candidate.shape or ref.dtype!=candidate.dtype or not torch.isfinite(ref).all():
                raise ValueError('reference vocabulary or dtype differs')
            refp=ref.float().log_softmax(-1).gather(-1,labels).squeeze(-1)
            delta=float((logp-refp).abs().max())
            if (bool(torch.equal(candidate,ref))!=c['exact_logits']
                    or float((candidate.float()-ref.float()).abs().max())!=c['max_logits_error']
                    or abs(delta-c['max_response_logp_error'])>1e-7):raise ValueError('reported comparison differs')
            if cell=='audio_padding':
                if delta>.1:raise ValueError('recomputed row exceeds unchanged threshold')
                errors.append(delta)
    return dict(container_id=CONTAINER,intent_sha256=INTENT,inspection_sha256=inspection_hash,
        result_sha256=file_hash(run/'result.json'),identity_sha256=file_hash(run/'identity.json'),
        adapter_sha256=identity['adapter_sha256'],
        rows=6,recomputed_reference_errors=errors,threshold=.1,
        scope='Complete frozen CPU candidate only; subsequent actual verl/microbatch/NPU checks remain required.')
