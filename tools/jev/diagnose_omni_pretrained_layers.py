#!/usr/bin/env python3
"""Full TRAIN6 layer tracing for the registered text-padding factor only.

Preflight prepares real audio inputs without loading weights. Full execution
requires a coordinator-captured successful terminal of the complete five-cell
diagnostic. No automatic dispatch/retry, generation, optimizer or training.
"""
import argparse
import importlib.metadata
import inspect
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_actor_batch import prepare, evidence
from multimodal_jev.omni_activation_trace import PaddingTrace
from diagnose_omni_pretrained_precision import cell_inputs
from analyze_omni_precision_logits import analyze
from probe_omni_pretrained_verl import read
from probe_omni_pretrained_forward import loading_report
from sample_sfc_omni_learning import load_inputs
from infer_qwen3_omni_probe import verify_assets


PRECISION_CONTAINER='0de5e886048040b9fa5489586f4a7fcddea88085701bd1818a8a6a94946e36db'
IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
IDENTITY_HASH='e39bdebee8d2d434df8a51efc9d57b4f697462f0a67e8577d72f7f2e97cae9dd'
INTENT_HASH='59eaf736137c18ea7a0a89485c5c663aeb610532fb576ba7886adc9d3d8ecb4a'
HF_HASH='a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340'


def pair_inputs(item,response,prompt_width,audio_width,pad_id):
    import torch
    pair=[]
    for text_pad in (False,True):
        x=cell_inputs(item,response,prompt_width=prompt_width,audio_width=audio_width,
                      fp32_features=True,audio_padding=True,text_padding=text_pad)
        x['input_ids'].masked_fill_(x['attention_mask']==0,pad_id)
        pair.append(x)
    left,right=pair
    for key in ('input_ids','position_ids'):
        if not torch.equal(left[key][left['attention_mask'].bool()],right[key][right['attention_mask'].bool()]):
            raise ValueError('valid text or position identity differs')
    for key in ('input_features','feature_attention_mask'):
        if not torch.equal(left[key],right[key]):raise ValueError('audio factor changed')
    return pair


def complete_dependency(root,inspection_path,inspection_hash):
    if not inspection_path or not inspection_hash or file_hash(inspection_path)!=inspection_hash:
        raise ValueError('coordinator-pinned terminal inspection required')
    state=read(inspection_path)
    if (state['Id']!=PRECISION_CONTAINER or state['Image']!=IMAGE
            or state['State']['Status']!='exited' or state['State']['ExitCode']!=0
            or state['State']['OOMKilled']):
        raise ValueError('exact successful complete precision container required')
    if read(root/'run/terminal.json')['exit_code']!=0:
        raise ValueError('precision application failed')
    launch=read(root/'launch.json');intent=read(root/'intent.json')
    if (launch['container_id']!=PRECISION_CONTAINER or launch['image']!=IMAGE
            or launch['intent_sha256']!=INTENT_HASH
            or file_hash(root/'intent.json')!=INTENT_HASH
            or launch['source_commit']!=intent['source_commit']):
        raise ValueError('precision launch identity differs')
    if (file_hash(root/'run/identity.json')!=IDENTITY_HASH
            or state['Config']['Cmd']!=intent['argv'][intent['argv'].index(IMAGE)+1:]
            or state['HostConfig']['NetworkMode']!='none'
            or state['HostConfig']['Devices'] or state['HostConfig']['Privileged']
            or state['HostConfig']['Memory']!=160*1024**3
            or state['HostConfig']['NanoCpus']!=16000000000):
        raise ValueError('precision command, resources or identity differs')
    for name,pin in intent['source_files'].items():
        if file_hash(Path(intent['source_root'])/name)!=pin:
            raise ValueError('original precision source snapshot changed')
    # Source snapshot was outside the output directory for this specific run.
    argv=intent['argv']
    runner=Path(argv[argv.index('python3')+2])
    if file_hash(runner)!=read(root/'run/identity.json')['runner_sha256']:
        raise ValueError('original precision runner changed')
    audit=analyze(root/'run',IDENTITY_HASH)
    result=read(root/'run/result.json')
    records=[read(root/'run'/name) for name in audit['record_sha256']]
    if (audit['completed_cells']!=30 or result['status']!='diagnostic_complete'
            or result['records']!=records or result['weight_updates']!=0):
        raise ValueError('complete thirty-condition evidence required')
    return dict(inspection_sha256=inspection_hash,result_sha256=file_hash(root/'run/result.json'),
                logits_audit=audit)


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor,Qwen3OmniMoeThinkerForConditionalGeneration
    torch.set_num_threads(16);torch.manual_seed(17);begin=time.monotonic()
    progress=lambda phase,**kw:atomic_json(args.output/'progress.json',dict(phase=phase,heartbeat_unix=time.time(),**kw))
    root=args.precision_root
    if file_hash(root/'run/identity.json')!=IDENTITY_HASH:raise ValueError('registered precision identity changed')
    original=read(root/'run/identity.json');cfg=original['config']
    base,rows,_,checks=load_inputs(cfg)
    if {k:importlib.metadata.version(k) for k in base['versions']}!=base['versions']:
        raise ValueError('fixed model runtime required')
    if file_hash(inspect.getfile(Qwen3OmniMoeThinkerForConditionalGeneration))!=HF_HASH:
        raise ValueError('fixed HF implementation required')
    dependency=None if args.preflight_only else complete_dependency(root,args.inspection,args.inspection_hash)
    progress('verify_assets');assets=verify_assets(base)
    if assets['files']!=original['model_files']:raise ValueError('model asset identity differs')
    processor=Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'],local_files_only=True)
    response=processor.tokenizer.encode('false',add_special_tokens=False)+[processor.tokenizer.eos_token_id]
    prepared=prepare(processor,rows,waveform_boundary='raw')
    for key,values in evidence(prepared).items():
        if any(checks[key][name]!=value for name,value in values.items()):raise ValueError('prompt changed')
    keys=[(r['id'],r['input']['decisions'][0]['id']) for r in rows]
    if (len(keys)!=6 or len(set(keys))!=6 or [list(k) for k in keys]!=original['keys']
            or response!=original['response_ids']):raise ValueError('full original six-row roster required')
    items=[prepared[k] for k in keys]
    pw=max(len(i['input_ids']) for i in items);aw=max(i['input_features'].shape[-1] for i in items)
    if (pw!=original['prompt_width'] or aw!=original['audio_width'] or pw+len(response)>4096
            or processor.tokenizer.pad_token_id is None):raise ValueError('original bounded input dimensions required')
    pairs=[pair_inputs(i,response,pw,aw,processor.tokenizer.pad_token_id) for i in items]
    atomic_json(args.output/'identity.json',dict(precision_identity_sha256=IDENTITY_HASH,keys=keys,
        source_sha256={str(Path(__file__).name):file_hash(__file__),
                       'tracer':file_hash(inspect.getfile(PaddingTrace)),
                       'cell_inputs':file_hash(inspect.getfile(cell_inputs))},
        model_files=assets['files'],response_ids=response,preflight_only=args.preflight_only,
        conditions=['audio_padding','text_padding'],dependency=dependency,
        text_padding_counts=[int(y['attention_mask'].numel()-y['attention_mask'].sum()) for x,y in pairs],
        waveform_boundary='raw',targets_read=False,weight_updates=0))
    if args.preflight_only:
        atomic_json(args.output/'result.json',dict(status='preflight_passed',rows=6,conditions=2,
                    model_loaded=False,weight_updates=0));return
    progress('load_thinker')
    model,loading=Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(base['audio_model'],
        local_files_only=True,dtype=torch.bfloat16,device_map='cpu',attn_implementation='sdpa',output_loading_info=True)
    loading=loading_report(loading);atomic_json(args.output/'loading-info.json',loading)
    if (any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs'))
            or any(not k.startswith(('talker.','code2wav.')) for k in loading.get('unexpected_keys',[]))):
        raise ValueError('complete Thinker required')
    model.eval().requires_grad_(False)
    if any(p.is_meta or p.device.type!='cpu' for p in model.parameters()):raise ValueError('materialized CPU model required')
    reports=[]
    for row,pair in enumerate(pairs):
        trace=PaddingTrace(args.output/f'row-{row:02d}-trace')
        replay=[]
        for index,(cell,inputs) in enumerate(zip(('audio_padding','text_padding'),pair,strict=True)):
            progress('forward',row=row,cell=cell)
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                with trace.capture(model.model,inputs['attention_mask'],compare=bool(index)):
                    output=model(**inputs)
                    logits=output.logits[0,-len(response)-1:-1].detach().clone()
                    del output
            reference=torch.load(root/'run'/f'row-{row:02d}-{cell}-logits.pt',map_location='cpu',weights_only=True)
            replay.append(dict(cell=cell,prior_logits_equal=bool(torch.equal(logits,reference)),
                               prior_logits_max_error=float((logits.float()-reference.float()).abs().max())))
            torch.save(logits,args.output/f'row-{row:02d}-{cell}-logits.pt')
        report=dict(row=row,key=keys[row],replay=replay,events=trace.reports,reference=trace.reference,
                    reference_tensor_bytes=trace.bytes,
                    first_difference=next((r['event'] for r in trace.reports if not r['equal']),None))
        atomic_json(args.output/f'row-{row:02d}.json',report);reports.append(report)
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('unexpected gradients')
    progress('verify_weights_after');verify_assets(base)
    atomic_json(args.output/'result.json',dict(status='diagnostic_complete',reports=reports,
        elapsed_seconds=time.monotonic()-begin,weight_updates=0,
        scope='All original TRAIN6 two-condition frozen layer traces; reproducibility differences retained, no parity acceptance or training.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('precision-root','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--inspection',type=Path);p.add_argument('--inspection-hash')
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))
