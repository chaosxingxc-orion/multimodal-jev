"""Full portable target client using the unchanged source-role/macro evaluator."""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import evaluate_context_transfer_macro as shared
from tools.reports import audit_context_transfer_macro as auditor
from tools.reports.context_transfer_portable_inputs import load

PINS={
    'evaluate_context_transfer_macro.py':'0221836966f54d7cec231aae68f9237d390ca738c28fc32486afa0593a329e80',
    'audit_context_transfer_macro.py':'04c48f9a3db57353e36c7fda3332be9372df0730fe2fdee1cdb51d18fafd428d',
}


def admitted(c):
    value=load(c['portable_full_inputs'])
    if value['dataset']!=c['dataset']:
        raise ValueError('portable full dataset differs')
    if value['dataset']=='SLURP_clean_TEST60' and value['source_release']!=c.get('source_release'):
        raise ValueError('portable clean TEST source ontology differs')
    return value


def configured(c):
    spec=c['portable_full_inputs']
    # This saved identity is audited separately, and carries no gold.
    identity={k:spec[k] for k in ('manifest_sha256','population_evidence_sha256')}
    if 'inputs' in c and c['inputs']!=identity:
        raise ValueError('portable client must not substitute historical target inputs')
    return dict(c,inputs=identity)


def bound(module,name):
    path=Path(module.__file__)
    if file_hash(path)!=PINS[path.name]:raise ValueError('shared portable evaluator changed')
    original=getattr(module,name)
    namespace=dict(vars(module),load=admitted)
    return FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)


def run(c,provenance):
    return bound(shared,'run')(configured(c),provenance)


def audit_outputs(c,collection,journal):
    c=configured(c)
    # The shared auditor checks input identities independently; additionally bind
    # which full portable admission was recorded by the actual client.
    evidence=json.loads((Path(collection)/'input-evidence.json').read_text())
    if evidence['inputs']!=c['inputs']:
        raise ValueError('saved portable full-population binding differs')
    return bound(auditor,'audit_outputs')(c,collection,journal)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    print(json.dumps(run(config,provenance),indent=2))
