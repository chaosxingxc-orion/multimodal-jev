"""GRPO provenance adapter for the shared replica selection packager.

Same model source, client, input-only panel and inference protocol. Only the
registered source curve and stage-specific runner guard differ from Stage1.
"""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import prepare_context_replica_selection as shared
from tools.reports.context_cached_grpo_source_plan import reconstruct


def inputs(c,provenance):
    plan=reconstruct(c)
    if c['parent']!=plan['source_operations']['training_package']['parent']:
        raise ValueError('same actual formal-training parent runtime required')
    fn=FunctionType(shared.inputs.__code__,dict(shared.inputs.__globals__,reconstruct_plan=lambda _:plan))
    return fn(c,provenance)


def runner(m):
    source=shared.runner(m)
    before="c['source_step'] not in [0,*range(32,318,32),318]"
    if source.count(before)!=1:raise ValueError('shared registered curve guard changed')
    source=source.replace(before,"c['source_step'] not in [0,*range(64,634,64),634]")
    compile(source,'<cached-grpo-selection-runner>','exec');return source


def prepare(c,provenance):
    def publish(path,value):
        if Path(path).name=='PREPARED.json':
            value=dict(value,status='replica_cached_grpo_selection_prepared_only',
                source_algorithm='dual_credit_grpo_floor_clipped_v1')
        commit_json(path,value)
    fn=FunctionType(shared.prepare.__code__,dict(shared.prepare.__globals__,inputs=inputs,runner=runner,commit_json=publish))
    result=fn(c,provenance)
    return dict(result,status='replica_cached_grpo_selection_prepared_only')


def audit(c,provenance):
    fn=FunctionType(shared.audit.__code__,dict(shared.audit.__globals__,prepare=prepare))
    result=fn(c,provenance)
    return dict(result,status='replica_cached_grpo_selection_package_audited_not_executed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],default='prepare');a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps((prepare if a.action=='prepare' else audit)(c,provenance),indent=2))
