"""Full cached-GRPO source curve using shared replica audit and transfer roles."""
import argparse
import inspect
import json
from types import FunctionType

from multimodal_jev.local_config import load_config
from tools.reports import collect_context_replica_infonce_curve as shared
from tools.reports import collect_context_macro_source_curve as publication
from tools.reports import context_cached_grpo_selection_operations as point
from tools.reports.context_cached_grpo_source_plan import reconstruct as reconstruct_plan


def reconstruct(c):
    source=inspect.getsource(shared.reconstruct)
    old="proof['status']!='complete_replica_infonce_selection_audited'"
    if source.count(old)!=1:raise ValueError('shared source point audit guard changed')
    source=source.replace(old,"proof['status']!='complete_replica_cached_grpo_selection_audited'")
    source=source.replace('complete_replica_infonce_curve','complete_replica_cached_grpo_curve')
    source=source.replace('registered InfoNCE curve','registered cached GRPO curve')
    scope=dict(shared.reconstruct.__globals__,reconstruct_plan=reconstruct_plan,point=point)
    exec(compile(source,'<cached-grpo-curve-reconstruction>','exec'),scope)
    return scope['reconstruct'](c)


def collect(c,provenance):
    fn=FunctionType(publication.collect.__code__,dict(publication.collect.__globals__,reconstruct=reconstruct))
    return fn(c,provenance)


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(collect(c,provenance),indent=2))
