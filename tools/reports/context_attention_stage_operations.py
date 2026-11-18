"""Versioned attention-boundary probe on the original ordered rollout32."""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.local_config import load_config
from tools.reports import context_audio_stage_operations as shared
from tools.reports.context_attention_stage_probe import compare,namespace

ADDITIONS=[*shared.ADDITIONS,'context_language_stage_probe.py','context_language_stage_capture.py',
    'context_language_stage_worker.py','context_attention_stage_capture.py',
    'context_attention_stage_probe.py','context_attention_stage_worker.py']


def argv(c):
    args=shared.argv(c)
    old=c['remote_root']+'/package/tools/reports/context_audio_stage_probe.py'
    args[args.index(old)]=c['remote_root']+'/package/tools/reports/context_attention_stage_probe.py'
    return args


def bind(fn):
    return FunctionType(fn.__code__,dict(fn.__globals__,ADDITIONS=ADDITIONS,argv=argv,validate=validate),fn.__name__)


def validate(c):
    namespace()
    receipt,config=bind(shared.validate)(c)
    if 'rollout_replay' not in c or len(config['rows'])!=32:
        raise ValueError('complete original ordered rollout32 required')
    return receipt,config


def prepare(c,provenance):return bind(shared.prepare)(c,provenance)


def audit(c,collection,handle):
    fn=FunctionType(shared.parent.audit.__code__,dict(shared.parent.audit.__globals__,validate=validate),'audit')
    result=fn(c,collection,handle)
    root=Path(collection)/'evaluation';actual=compare(root,validate(c)[1])
    if actual!=json.loads((root/'report.json').read_text()):raise ValueError('attention tensor replay differs')
    return dict(result,status='attention_boundaries_independently_compared_not_admission')


def launch(c,provenance):
    ns=shared.parent.namespace();ns.update(validate=validate,audit=audit)
    return ns['launch'](c,provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','launch'],required=True);a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else validate(c)[0] if a.action=='validate' else launch(c,provenance)
    print(json.dumps(result,indent=2))
