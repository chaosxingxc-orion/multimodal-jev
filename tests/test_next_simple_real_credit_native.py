"""Fixed-image check on authenticated recorded TRAIN actions; no rollout/model/device requests."""
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.objective_plugin import ObjectivePlugin


def test_actual_behavior_drop_set_credit_and_keep_other_logit_gradients():
    root=Path(__file__).resolve().parents[1]
    design=json.loads((root/'configs/experiments/stage2-without-set-credit-v1.json').read_bytes())
    parent=json.loads((root/'configs/experiments/stage2-probability-margin-candidate-v0.0.126.json').read_bytes())
    path=root/design['objective_source']
    assert file_hash(path)==design['objective_sha256']
    old,new=parent['options'],design['options']
    assert set(old)==set(new) and {k for k in old if old[k]!=new[k]}=={'task_rl_weight'}
    assert old['task_rl_weight']==.5 and new['task_rl_weight']==0.
    group=torch.load(root/'fixtures/formal-first.pt',weights_only=True,map_location='cpu')
    rows=[r for r in group if r['real_request']][:8]  # first contiguous real microbatch, no outcome selection
    assert len(rows)==8
    records=[r['dual_credit'] for r in rows]
    ids=[r['dual_identity'] for r in rows]
    ref=torch.stack([r['reference_probability'] for r in rows])
    z=torch.tensor([r['binding']['logits'] for r in records],requires_grad=True)
    def gradient(opts):
        plugin=ObjectivePlugin(dict(path=path,sha256=design['objective_sha256'],options=opts))
        loss,m=plugin.hybrid_loss(z,records,ids,ref,torch.ones(len(rows),dtype=torch.bool),
                                 global_real_requests=len(rows))
        return torch.autograd.grad(loss,z)[0],m
    a,ma=gradient(old); b,mb=gradient(new)
    isolated=dict(old)
    for key in ('ranking_rl_weight','existence_rl_weight','ranking_auxiliary_coefficient',
                'probability_nll_sum_coefficient','reference_kl_mean_coefficient','probability_margin_coefficient'):
        isolated[key]=0.
    g,_=gradient(isolated)
    assert g.norm()>0 and mb['task_rl']==0.
    torch.testing.assert_close(a-b,g,rtol=3e-5,atol=2e-7)
    assert all(ma[k]==mb[k] for k in ('ranking_rl','existence_rl','auxiliary','kl','nll_sum','hardest_pair'))
