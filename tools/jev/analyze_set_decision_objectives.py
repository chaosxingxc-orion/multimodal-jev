#!/usr/bin/env python3
"""Analytic finite-support objective diagnostics, never a production trainer."""
import argparse
import json
from pathlib import Path
import torch


def objectives(logits, acceptable):
    if (logits.ndim != 1 or not logits.is_floating_point() or len(logits) < 2
            or not torch.isfinite(logits).all() or acceptable.dtype != torch.bool
            or acceptable.shape != logits.shape or acceptable.device != logits.device
            or not acceptable.any() or acceptable.all()):
        raise ValueError('finite support with known nonempty valid and invalid sets required')
    logp = logits.log_softmax(-1)
    mass = logp[acceptable].logsumexp(0).exp()
    # Tau=1, margin=0 are diagnostic coordinates, not selected hyperparameters.
    pair = torch.nn.functional.softplus(
        logits[~acceptable][None, :] - logits[acceptable][:, None]).mean()
    return dict(set_mass_nll=-logp[acceptable].logsumexp(0),
        uniform_valid_ce=-logp[acceptable].mean(), pairwise_valid_invalid=pair,
        negative_expected_membership_reward=-mass)


def evaluate(probabilities, valid):
    logits = torch.tensor(probabilities, dtype=torch.float64).log().requires_grad_()
    mask = torch.tensor(valid, dtype=torch.bool)
    values = objectives(logits, mask)
    gradients = {k: torch.autograd.grad(v, logits, retain_graph=True)[0].tolist()
                 for k, v in values.items()}
    mass = float(logits.softmax(-1)[mask].sum().detach())
    return dict(probabilities=probabilities, acceptable=valid, acceptable_mass=mass,
        losses={k:float(v.detach()) for k,v in values.items()}, logit_gradients=gradients,
        G16_constant_membership_reward_probability=(mass**16+(1-mass)**16),
        interpretation='Uniform-valid CE assumes an extra target distribution; pairwise requires every valid to outrank every invalid. Set mass and membership reward only require choosing any valid.')


def report():
    cases = {
        'balanced_valid': ([.45,.45,.10], [True,True,False]),
        'concentrated_valid': ([.89,.01,.10], [True,True,False]),
        'poor_valid_mass': ([.05,.05,.90], [True,True,False]),
        'one_valid': ([.70,.20,.10], [True,False,False]),
        'uniform_60_one_valid': ([1/60]*60, [True]+[False]*59),
        'uniform_7_two_valid': ([1/7]*7, [True,True]+[False]*5),
    }
    results={name:evaluate(*case) for name,case in cases.items()}
    # Equal Final mass with different independent applicability calibration.
    truth=torch.tensor([1.,1.,0.],dtype=torch.float64)
    applicability={}
    for name,p in [('both_supported',[.95,.95,.05]),('miss_second',[.95,.05,.05])]:
        q=torch.tensor(p,dtype=torch.float64)
        applicability[name]=dict(probabilities=p,mean_binary_brier=float(((q-truth)**2).mean()))
    return dict(schema='set-decision-objective-diagnostic-v1',cases=results,
        independent_applicability=applicability,production_loss_changed=False,training_dispatch_allowed=False,
        scope='Exact FP64 finite candidate algebra and gradients, no model predictions, native verl execution, NPU, or task efficacy. G16 uses iid membership reward and no process reward.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=report();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:dict(mass=v['acceptable_mass'],losses=v['losses'],constant_G16=v['G16_constant_membership_reward_probability']) for k,v in result['cases'].items()},indent=2))
