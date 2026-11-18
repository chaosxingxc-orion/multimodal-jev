"""Replay pinned finite objective traces; no model calls or NPU dispatch.

Group hashes must come from the independent collection. This supplements,
but does not replace, source/rollout/truth/reference/optimizer auditing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.objective_plugin import ObjectivePlugin
from multimodal_jev.objective_trace import audit_update


def checked(binding):
    if set(binding)!={'path','sha256'}:raise ValueError('explicit file and SHA256 required')
    p=Path(binding['path'])
    if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=binding['sha256']:
        raise ValueError('trace/group evidence changed: '+str(p))
    return p


def audit(c,provenance):
    c=dict(c)
    replay_device=c.pop('replay_device','cpu')
    if replay_device not in ('cpu','npu:0'):
        raise ValueError('explicit CPU or isolated NPU replay device required')
    if replay_device=='npu:0':
        import torch_npu
        torch.npu.set_device(0)
    if (set(c)!={'objective_plugin','world_size','local_batch','microbatch','updates','loss_atol','output'}
            or any(type(c[k]) is not int or c[k]<1 for k in ('world_size','local_batch','microbatch'))
            or c['local_batch']%c['microbatch'] or not isinstance(c['updates'],list)
            or not 1<=len(c['updates'])<=2):
        raise ValueError('explicit finite two-update trace configuration required')
    plugin=ObjectivePlugin(c['objective_plugin']);reports=[];previous=None;paths=set()
    for item in c['updates']:
        if (set(item)!={'update','groups','traces'} or type(item['update']) is not int
                or item['update']<0 or previous is not None and item['update']!=previous+1
                or len(item['traces'])!=c['world_size']):
            raise ValueError('consecutive updates and complete ordered ranks required')
        previous=item['update'];groups_path=checked(item['groups'])
        if groups_path.resolve() in paths:raise ValueError('duplicate evidence path')
        paths.add(groups_path.resolve())
        groups=torch.load(groups_path,map_location='cpu',weights_only=True)
        if not isinstance(groups,list) or len(groups)!=c['world_size']*c['local_batch']:
            raise ValueError('exact declared global group population required')
        ranks=[groups[i:i+c['local_batch']] for i in range(0,len(groups),c['local_batch'])]
        docs=[]
        for rank,binding in enumerate(item['traces']):
            if set(binding)!={'rank','path','sha256'} or type(binding['rank']) is not int or binding['rank']!=rank:
                raise ValueError('explicit rank order required')
            p=checked({k:binding[k] for k in ('path','sha256')})
            if p.resolve() in paths:raise ValueError('duplicate evidence path')
            paths.add(p.resolve());docs.append(json.loads(p.read_text()))
        reports.append(audit_update(docs,ranks,plugin,update=item['update'],
                                    microbatch=c['microbatch'],loss_atol=c['loss_atol'],replay_device=replay_device))
    return dict(status='finite_objective_traces_replayed',objective=plugin.identity,
        updates=reports,evidence=c['updates'],provenance=provenance,
        scope='Pinned group-to-trace coverage and objective replay only; source, truth, rollout, reference and optimizer audit remain separate',
        replay_device=replay_device,training_dispatch_allowed=False,npu_admitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--local',required=True);args=p.parse_args()
    c,provenance=load_config(args.config,args.local)
    result=audit(c,provenance);commit_json(c['output'],result)
    print(json.dumps(result,indent=2))
