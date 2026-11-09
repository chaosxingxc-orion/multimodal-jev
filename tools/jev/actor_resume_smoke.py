#!/usr/bin/env python3
"""Compare uninterrupted vs saved/reloaded real LoRA second optimizer step.

Uses the prior controlled-fixture actor, not sampled vLLM trajectories.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from peft import PeftModel
from multimodal_jev.asr import load_runtime
from multimodal_jev.actor_state import save_optimizer,restore_optimizer,trainable_parameters
from multimodal_jev.audio_policy import teacher_forcing,load_audio
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.task_schema import JevExample


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--fixture-run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);start=time.monotonic()
        prior=json.loads((a.fixture_run/'result.json').read_text())
        assert prior['status']=='passed' and json.loads((a.fixture_run/'terminal.json').read_text())['exit_code']==0
        with a.manifest.open() as f:
            ex=next(JevExample.from_record(row) for line in f if (row:=json.loads(line))['metadata']['variant']=='noul.True')
        assert ex.split=='train' and file_hash(ex.audio)==ex.metadata['audio_sha256']
        def load():
            proc,base=load_runtime(a.model,'cpu','float32')
            model=PeftModel.from_pretrained(base,a.fixture_run/'adapter',is_trainable=True).eval()
            optim=torch.optim.AdamW([p for _,p in trainable_parameters(model)],lr=1e-4,foreach=False,fused=False)
            return proc,model,optim
        proc,model,optim=load()
        # This legacy fixture state came from the exact same q/v LoRA named order.
        optim.load_state_dict(torch.load(a.fixture_run/'optimizer.pt',map_location='cpu',weights_only=True))
        assert all(int(s['step'])==1 for s in optim.state.values())
        save_optimizer(model,optim,a.output/'resume.pt')
        batch,_=teacher_forcing(proc,ex,load_audio(ex.audio))
        def step(model,optim):
            optim.zero_grad(set_to_none=True)
            loss=model(**batch,use_cache=False).loss
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_([p for _,p in trainable_parameters(model)],1.,error_if_nonfinite=True)
            optim.step()
            return float(loss.detach())
        atomic_json(a.output/'progress.json',{'phase':'uninterrupted_step','heartbeat_unix':time.time()})
        first_loss=step(model,optim)
        expected={name:p.detach().clone() for name,p in trainable_parameters(model)}
        expected_state={name:{k:v.detach().clone() if torch.is_tensor(v) else v for k,v in optim.state[p].items()}
                        for name,p in trainable_parameters(model)}
        del model,optim;gc.collect()
        proc,model,optim=load()
        restore_optimizer(model,optim,a.output/'resume.pt')
        assert all(int(s['step'])==1 for s in optim.state.values())
        atomic_json(a.output/'progress.json',{'phase':'resumed_step','heartbeat_unix':time.time()})
        second_loss=step(model,optim);errors=[]
        for name,param in trainable_parameters(model):
            torch.testing.assert_close(param,expected[name],atol=1e-7,rtol=1e-6)
            errors.append(float((param-expected[name]).abs().max().detach()))
            for key,value in optim.state[param].items():
                torch.testing.assert_close(value,expected_state[name][key],atol=1e-7,rtol=1e-6)
            assert int(optim.state[param]['step'])==2
        report={'status':'passed','parameters':len(errors),'max_parameter_error':max(errors),
                'uninterrupted_loss':first_loss,'resumed_loss':second_loss,'optimizer_step':2,
                'optimizer_moments_match':True,'elapsed_seconds':time.monotonic()-start,
                'helper_sha256':file_hash(Path(__file__).parents[2]/'src/multimodal_jev/actor_state.py'),
                'scope':'real model/adapter and AdamW save-load continuation; controlled teacher-forcing step, not on-policy RL or efficacy'}
        atomic_json(a.output/'result.json',report);atomic_json(a.output/'terminal.json',{'exit_code':0})
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',{'exit_code':1,'error':repr(exc)});raise


if __name__=='__main__':main()
