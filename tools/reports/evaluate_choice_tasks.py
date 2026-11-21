"""Compare frozen policies with the same Stage1 proposals and label-free typed inputs."""
from pathlib import Path
import torch
from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_choice import distribution, shortlist, lengths, varied_slate
from multimodal_jev.context_policy_snapshot import load_policy
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.prepare_native_seal_cache import plan
from tools.reports.run_native_seal_endpoint import read,bound


def evaluate(spec,c,parent,model,device,processor,collate,rows,sources,provider):
    policies=with_baseline(spec,sources)
    criterion=spec['choice']
    assets={r['id']:req for _,r,req in plan(c['cache'])}
    compiler=provider.choice_compiler(processor,collate,parent['view_execution']['max_input_tokens'])
    width=spec['batch']
    proposals={}
    profiles={}
    paired={}
    def forward(batch,pid):
        return provider.forward(model,batch,device=device,dtype=torch.bfloat16,policy_id=pid,**collate).float()
    def input_row(row,uid,pid):return dict(row,policy_id=pid,request_id=uid)
    if spec.get('full_ontology_only'):
        with torch.inference_mode():
            for arm,(path,pid) in policies.items():
                if processor_cache_identity(path)!=processor_cache_identity(c['snapshot']):
                    raise ValueError('incompatible Choice processor')
                load_policy(path,model,expected_policy_id=pid,expected_contract=c['contract'],expected_base_id=c['base_id'])
                records=[]
                for start in range(0,len(rows),width):
                    group=[]
                    for asset,row in rows[start:start+width]:
                        uid=asset['id'];req=assets[uid];x=input_row(row,uid,pid)
                        compiler.render(x,req,'open',list(range(len(req.questions))))
                        group.append(x)
                    logp=distribution(forward(group,pid),'open',criterion['temperature']).cpu()
                    if not torch.isfinite(logp).all():raise ValueError('invalid action log probability')
                    records.extend(dict(id=x['request_id'],logp=v) for x,v in zip(group,logp.tolist(),strict=True))
                commit_json(Path(spec['output'])/(arm+'-choices.json'),dict(policy_id=pid,records=records))
        return
    evaluate_shortlists(spec,c,model,processor,rows,policies,criterion,assets,compiler,width,proposals,profiles,paired,forward,input_row)


def with_baseline(spec,sources):
    if 'choice_baseline' not in spec:
        if not spec.get('full_ontology_only') or 'stage1' not in sources:
            raise ValueError('missing comparable Choice baseline')
        return sources
    spec_baseline=spec['choice_baseline']
    baseline=Path(spec_baseline['root'])
    receipt=bound(dict(path=baseline/'receipt.json',sha256=spec_baseline['receipt_sha256']))
    if (not receipt['passed'] or receipt['window']!='formal' or
            file_hash(baseline/'policy/MANIFEST.json')!=receipt['manifest_sha256']):
        raise ValueError('invalid comparison endpoint')
    return dict(sources,prior=(str(baseline/'policy'),receipt['endpoint']))

def evaluate_shortlists(spec,c,model,processor,rows,policies,criterion,assets,compiler,width,proposals,profiles,paired,forward,input_row):
    with torch.inference_mode():
        for start in range(0,len(rows),width):
            batch=[input_row(row,asset['id'],c['policy_id']) for asset,row in rows[start:start+width]]
            prob=forward(batch,c['policy_id']).sigmoid().cpu().tolist()
            for item,scores in zip(batch,prob,strict=True):
                uid=item['request_id']
                ids=[q.id for q in assets[uid].questions]
                if 'paired_lengths' in spec:
                    paired[uid]={k:varied_slate(scores,ids,uid,criterion,k,'evaluation')
                        for k in lengths(dict(candidate_lengths=spec['paired_lengths']),len(ids))}
                if 'candidate_lengths' in criterion:
                    widths=lengths(criterion,len(ids))
                    # All items in one inference batch have the same marker count.
                    bucket=((start//width)*spec['shards']+spec['shard']) % len(widths)
                    proposals[uid],profiles[uid]=varied_slate(scores,ids,uid,criterion,widths[bucket],'evaluation')
                else:
                    proposals[uid]=shortlist(scores,ids,uid,criterion)
        for arm,(path,pid) in policies.items():
            if processor_cache_identity(path)!=processor_cache_identity(c['snapshot']):
                raise ValueError('incompatible Choice processor')
            load_policy(path,model,expected_policy_id=pid,expected_contract=c['contract'],expected_base_id=c['base_id'])
            records=[]
            for start in range(0,len(rows),width):
                batch=rows[start:start+width]
                records_here=[dict(id=a['id'],slate=[assets[a['id']].questions[i].id for i in proposals[a['id']]]) for a,_ in batch]
                if profiles:
                    for rec in records_here:rec['proposal']=profiles[rec['id']]
                for mode in ('closed','open'):
                    group=[]
                    for asset,row in batch:
                        uid=asset['id'];req=assets[uid];x=input_row(row,uid,pid)
                        indices=proposals[uid] if mode=='open' else list(range(len(req.questions)))
                        compiler.render(x,req,'open' if profiles else mode,indices)
                        group.append(x)
                    logits=forward(group,pid)
                    if profiles and mode=='closed':logits=logits[:,:-1]
                    values=distribution(logits,mode,criterion['temperature']).exp().cpu().tolist()
                    for rec,v in zip(records_here,values,strict=True):rec[mode]=v
                if paired:
                    # Reuse exactly the same recordings and nested candidate ordering at each K.
                    for rec in records_here:rec['panels']={}
                    for k in paired[records_here[0]['id']]:
                        group=[]
                        for asset,row in batch:
                            uid=asset['id'];req=assets[uid];x=input_row(row,uid,pid)
                            indices,kind=paired[uid][k]
                            compiler.render(x,req,'open',indices);group.append(x)
                        values=distribution(forward(group,pid),'open',criterion['temperature']).exp().cpu().tolist()
                        for rec,v in zip(records_here,values,strict=True):
                            uid=rec['id'];indices,kind=paired[uid][k]
                            rec['panels'][str(k)]=dict(slate=[assets[uid].questions[i].id for i in indices],
                                                      proposal=kind,open=v)
                records.extend(records_here)
            commit_json(Path(spec['output'])/(arm+'-choices.json'),dict(policy_id=pid,records=records))
