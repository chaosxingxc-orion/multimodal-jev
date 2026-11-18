"""Offline diagnosis from actual failed macro outputs, without new inference."""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.local_config import load_config
from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_portable_release import load
from multimodal_jev.context_training_schedule import load_schedule
from multimodal_jev.context_dual_credit import prepare,verify,ALGORITHMS,components,log_probabilities


def run(c):
    package=Path(c['package']);root=Path(c['failure_collection']);t=json.loads((package/'training.json').read_text())
    hashes=json.loads(Path(c['failure_files']).read_text())
    if {str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()}!=hashes:raise ValueError('collected failure evidence changed')
    rows,task,_=load(t['release'],t['release_sha256'],'learning',purpose='optimize');population={r['id']:r for r in rows}
    batches,_=load_schedule(package/'schedule',t['schedule_sha256'],release_sha256=t['schedule_release_sha256'],population=population)
    directory=root/'journal'/t['macro_client']['attempt_id']/'batch-00000001'
    manifest=json.loads((directory/'INPUT.json').read_text());records=[None]*128
    for job in manifest['jobs']:
        for i,r in zip(job['indices'],json.loads((directory/(job['rollout_id']+'.json')).read_text())['records'],strict=True):records[i]=r
    from tools.reports.audit_context_macro_training import audit_journal
    audit_journal(directory,records,policy=t['source_policy_id'],client=t['macro_client'])
    ids=[v['id'] for v in task['schema']['candidates']];sat=0;maxdelta=0.;existence_gap=0.
    for i,(slot,r) in enumerate(zip(batches[0]['slots'],records,strict=True)):
        z=torch.tensor(r['logits']);p=torch.tensor(r['prob']);y=torch.tensor([k==population[slot['id']]['intent'] for k in ids])
        idx=sorted(range(len(ids)),key=lambda i:ids[i]);pairs=[(a,b) for n,a in enumerate(idx) for b in idx[n+1:] if y[a]!=y[b]]
        delta=torch.stack([z[a]-z[b] for a,b in pairs]);q=delta.sigmoid();sat+=int(((q<=0)|(q>=1)).sum());maxdelta=max(maxdelta,float(delta.abs().max()))
        identity=dict(policy_id=r['policy_id'],input_id=r['request_id'],query_id=slot['id']+'::'+str(i),update=0)
        credit=prepare(z,p,ids,y,**identity,algorithm=ALGORITHMS[1]);verify(credit)
        # Existence must be recomputed on the actual NPU actor, not declared
        # equal from a CPU sigmoid. Keep that backend gap visible.
        existence_gap=max(existence_gap,float((log_probabilities(z.sigmoid())-log_probabilities(p)).abs().max()))
        ranking=components(z.requires_grad_(),ids,y)['ranking']
        assert torch.equal(ranking.probability.detach(),credit['branches']['ranking']['probability'])
        grad,=torch.autograd.grad(log_probabilities(ranking.probability).sum(),z)
        if not torch.isfinite(grad).all():raise ValueError('nonfinite ranking gradient')
    return dict(rows=128,old_fp32_saturated_ranking_components=sat,max_absolute_pair_logit=maxdelta,
        new_pair_dtype='float64',all_credit_recomputed=True,ranking_probability_and_gradient_checked=True,
        cpu_vs_actual_npu_existence_max_logprob_gap=existence_gap,actor_npu_verified=False,
        source_files=hashes,scope='Offline actual engine records and scheduled labels; no NPU actor/update or efficacy proof.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local);torch.set_num_threads(1);result=run(c)
    with Path(c['output']).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'}))
