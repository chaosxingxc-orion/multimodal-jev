"""Compare genuine HF, serial-acoustic and batched-acoustic model activations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hashlib


def load_rows(config):
    import torch
    from multimodal_jev.span_request_identity import tensor_identity
    result = []
    for row in config['rows']:
        path = Path(config['cache_root'])/(row['cache_key']+'.pt')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['cache_sha256']:
            raise ValueError('cache identity differs')
        data = torch.load(path, map_location='cpu', weights_only=True)
        actual = {k: tensor_identity(v) for k,v in dict(data['inputs'], positions=data['positions']).items()}
        if actual != row['tensors']: raise ValueError('input tensors differ')
        result.append(dict(inputs=data['inputs'], positions=data['positions']))
    return result


def phase(package, output, mode):
    import torch
    import torch_npu
    from multimodal_jev.local_checkpoint import commit_json, commit_file
    c = json.loads((package/'config.json').read_text()); output.mkdir(exist_ok=False)
    rows = load_rows(c); owner = None
    torch.set_num_threads(2); torch.npu.set_device(0); torch.manual_seed(317); torch.npu.manual_seed_all(317)
    try:
        if mode == 'hf':
            from multimodal_jev.asr import load_runtime
            from multimodal_jev.sft import initialize_adapter
            from multimodal_jev.context_decision_head import ContextDecisionHead
            from multimodal_jev.context_decision_model import ContextDecisionModel
            from multimodal_jev.context_policy_snapshot import load_policy
            from multimodal_jev.context_runtime_profile import PROFILE, install_actor
            from tools.reports.context_audio_stage_capture import attach
            p = c['policy']; device = torch.device('npu', 0)
            _, base = load_runtime(p['model'], device, 'bfloat16')
            adapted = initialize_adapter(base, dict(lora_rank=4, lora_alpha=8, lora_execution_dtype='bfloat16')).eval()
            head = ContextDecisionHead(1024,8,1024,.1).to(device=device,dtype=torch.bfloat16).eval()
            model = ContextDecisionModel(adapted.get_base_model().model,head,p['contract']['token']['id']).eval()
            load_policy(p['snapshot'], model, expected_policy_id=p['policy_id'], expected_contract=p['contract'], expected_base_id=p['base_id'])
            handle, runtime = install_actor(adapted, model, PROFILE)
            commit_json(output/'runtime.json', runtime)
            records = []
            for i, row in enumerate(rows):
                hooks, capture = attach(base.model.audio_tower, base.model.multi_modal_projector)
                with torch.no_grad():
                    z = model({k:v.to(device) for k,v in row['inputs'].items()},row['positions'].to(device),return_logits=True).float()[0]
                for hook in hooks: hook.remove()
                commit_file(output/f'row-{i:02d}.pt',lambda f:torch.save(capture,f))
                records.append(dict(logits=z.cpu().tolist(),prob=z.sigmoid().cpu().tolist()))
        else:
            import tools.reports.context_acoustic_owner_candidate as original
            source = Path(original.__file__).read_text()
            source = source.replace("worker='AcousticBatchWorker' if acoustic else 'AcousticBaselineWorker'", "worker='StageAcousticWorker' if acoustic else 'StageBaselineWorker'")
            source = source.replace("'tools.reports.context_acoustic_worker_candidate.'+worker", "'tools.reports.context_audio_stage_worker.'+worker")
            ns={'__name__':__name__+'.owner','__file__':original.__file__}
            exec(compile(source,'<stage-owner>','exec'), ns)
            owner=ns['owner_class'](str(package),acoustic=mode=='acoustic')(c['policy'])
            rpc=owner.engine.llm_engine.collective_rpc
            records=[]
            for i,row in enumerate(rows):
                rpc('jev_stage_capture',args=('start',''))
                records.append(owner.score_many([row],audio_token_id=151676)[0])
                rpc('jev_stage_capture',args=('finish',str(output/f'row-{i:02d}.pt')))
            # New UUIDs force actual acoustic work after individual requests.
            from tools.reports.measure_context_acoustic_candidate import fresh_audio_pass
            rpc('jev_stage_capture',args=('start',''))
            with fresh_audio_pass(owner,'stage-batch'):
                batched=owner.score_many(rows,audio_token_id=151676)
            rpc('jev_stage_capture',args=('finish',str(output/'batch.pt')))
            commit_json(output/'batch-records.json',batched)
            owner.close(); owner=None
        commit_json(output/'records.json',records)
        commit_json(output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    finally:
        if owner is not None: owner.close()


def compare(output, config):
    import torch
    from tools.reports.context_acoustic_numeric_probe import difference
    def read(mode,name):return torch.load(output/mode/name,map_location='cpu',weights_only=True)
    def flat(parts):return torch.cat([v.reshape(-1,v.shape[-1]) for v in parts],dim=0)
    reports=[]; references=[read('hf',f'row-{i:02d}.pt') for i in range(len(config['rows']))]
    for i,row in enumerate(config['rows']):
        ref=references[i]; by_mode={}
        for mode in ('baseline','acoustic'):
            candidate=read(mode,f'row-{i:02d}.pt'); stages={}
            for name in ref:
                a,b=flat(ref[name]),flat(candidate[name])
                stages[name]=difference(a,b) if a.shape==b.shape and a.dtype==b.dtype else dict(shape_mismatch=[list(a.shape),list(b.shape)],dtypes=[str(a.dtype),str(b.dtype)])
            by_mode[mode]=stages
        reports.append(dict(id=row['id'],comparisons=by_mode))
    batch={}
    for mode in ('baseline','acoustic'):
        captured=read(mode,'batch.pt');stages={}
        for name in ('encoder_input',*[f'layer_{i:02d}' for i in range(18)],'ln_post','projected'):
            a=torch.cat([flat(ref[name]) for ref in references])
            b=flat(captured[name])
            stages[name]=difference(a,b) if a.shape==b.shape else dict(shape_mismatch=[list(a.shape),list(b.shape)])
        batch[mode]=stages
    result=dict(status='actual_model_stage_comparison',rows=reports,batch=batch,full_model_parity_verified=False,training_dispatch_allowed=False)
    if (output/'hf/records.json').exists():
        expected=json.loads((output/'hf/records.json').read_text());scores={}
        for mode in ('baseline','acoustic'):
            scores[mode]={}
            for part in ('records','batch-records'):
                actual=json.loads((output/mode/(part+'.json')).read_text())
                if len(actual)!=len(expected):raise ValueError('complete output records required')
                scores[mode][part]=[dict(id=row['id'],logits=difference(torch.tensor(a['logits']),torch.tensor(b['logits'])),
                    prob=difference(torch.tensor(a['prob']),torch.tensor(b['prob'])))
                    for row,a,b in zip(config['rows'],expected,actual,strict=True)]
        result['scores']=scores
    return result


def run(package,output):
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    from tools.reports.run_context_batch_probe_candidate import verify
    require_training_storage(output);output.mkdir(exist_ok=False)
    try:
        manifest=verify(package);commit_json(output/'source.json',dict(files_sha256=manifest))
        for mode in ('hf','baseline','acoustic'):
            with (output/(mode+'.log')).open('x') as log:
                proc=subprocess.run([sys.executable,'-B',__file__,'--package',str(package),'--output',str(output/mode),'--phase',mode],stdout=log,stderr=subprocess.STDOUT)
            commit_json(output/(mode+'-exit.json'),dict(exit_code=proc.returncode))
            if proc.returncode:raise RuntimeError(mode+' capture failed')
        report=compare(output,json.loads((package/'config.json').read_text()))
        commit_json(output/'report.json',report);commit_json(output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['hf','baseline','acoustic']);a=p.parse_args()
    if a.phase:phase(a.package,a.output,a.phase)
    else:run(a.package,a.output)
