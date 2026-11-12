#!/usr/bin/env python3
"""Actual pinned verl loss/gradient checks for categorical finite credit."""
import argparse
import inspect
from pathlib import Path
import subprocess
import sys
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_finite_credit import reference_loss,verl_loss


def run():
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    torch.set_num_threads(2);torch.manual_seed(317)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    cases=[]
    for widths in ((60,),(60,2,5),(2,2,2,2,2,2)):
        for shift in (0.,.1,1.,8.,30.):
            base=[torch.randn(3,k) for k in widths]
            logits=[(v+torch.linspace(-shift,shift,v.shape[1])).requires_grad_() for v in base]
            old=tuple(v.log_softmax(-1) for v in base);actor=tuple(v.log_softmax(-1) for v in logits)
            targets=torch.tensor([[k-1 for k in widths],[0 for k in widths],[-100 for k in widths]])
            mask=targets!=-100;weights=torch.full_like(targets,.1,dtype=torch.float32);weights[:,0]=1
            rw=torch.tensor([1.,3.,2.])
            actual,metrics=verl_loss(old,actor,targets,mask,weights,rw)
            expected=reference_loss(old,actor,targets,mask,weights,rw)
            a=torch.autograd.grad(actual,logits,retain_graph=True);b=torch.autograd.grad(expected,logits)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
            for x,y in zip(a,b,strict=True):torch.testing.assert_close(x,y,atol=1e-6,rtol=0)
            cases.append(dict(candidate_sizes=widths,shift=shift,loss=float(actual.detach()),
                gradient_max_error=max(float((x-y).abs().max()) for x,y in zip(a,b,strict=True)),metrics=metrics))
    return dict(passed=True,cases=cases,core_source_sha256=file_hash(inspect.getfile(compute_policy_loss_vanilla)),
        scope='Synthetic FP32 CPU actual verl marginal finite-support PPO loss/gradients; no audio/NPU/native actor integration or efficacy evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        tests=Path(__file__).resolve().parents[2]/'tests/test_span_finite_credit.py'
        checked=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(tests)],capture_output=True,text=True)
        atomic_json(a.output/'tests.json',dict(returncode=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
        checked.check_returncode();atomic_json(a.output/'result.json',run());atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
