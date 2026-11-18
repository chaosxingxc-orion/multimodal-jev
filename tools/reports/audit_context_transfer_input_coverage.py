"""Bind the admitted parity subset to portable full evaluation inputs.

Input-only proof; does not certify actual model/backend parity or cached bytes.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_macro_rollout import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_transfer_portable_inputs import load as load_full
from tools.reports.context_transfer_parity_inputs import load as load_panel


def audit(full_spec,panel_spec):
    full=load_full(full_spec);panel=load_panel(panel_spec);selected=panel['selected_panel']
    if (full['dataset']!=panel['dataset'] or full['candidate_ids']!=panel['candidate_ids']
            or full['processor_sha256']!=panel['processor_sha256']
            or selected['population_sha256']!=digest(full['population'])
            or selected['input_receipts_sha256']!=digest(full['input_receipts'])):
        raise ValueError('parity panel must originate from the exact full target inputs')
    indices=selected['indices']
    if (any(type(i) is not int or not 0<=i<len(full['population']) for i in indices)
            or indices!=sorted(set(indices))
            or [full['population'][i] for i in indices]!=panel['population']
            or [full['input_receipts'][i] for i in indices]!=panel['input_receipts']):
        raise ValueError('parity subset must match full evaluation population order and tensors')
    for i in indices:
        key=full['population'][i]['id']
        if full['requests'][key]!=panel['requests'][key]:
            raise ValueError('full and parity requests differ')
    return dict(status='portable_full_and_parity_inputs_bound',dataset=full['dataset'],
        recordings=len(full['population']),parity_recordings=len(indices),decisions=len(full['candidate_ids']),
        full_manifest_sha256=full_spec['manifest_sha256'],panel_manifest_sha256=panel_spec['manifest_sha256'],
        population_evidence_sha256=full_spec['population_evidence_sha256'],
        panel_sha256=panel_spec['panel_sha256'],processor_sha256=full['processor_sha256'],
        cache_bytes_verified=False,npu_executed=False,
        scope='All portable inputs match independent full population evidence; selected parity inputs '
              'are an exact subset. No cache bytes, model output or full-population parity claim.')


def run(c,provenance):
    output=Path(c['output'])
    if not output.is_absolute() or output.exists():raise ValueError('new absolute coverage report required')
    for spec in (c['full'],c['panel']):
        if output.resolve().is_relative_to(Path(spec['root']).resolve()):
            raise ValueError('coverage report cannot modify sealed inputs')
    result=audit(c['full'],c['panel']);result['config_provenance']=provenance
    check_paths({'coverage_report':output});output.parent.mkdir(parents=True,exist_ok=True)
    commit_json(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();c,provenance=load_config(args.config,args.local)
    print(json.dumps(run(c,provenance),indent=2))
