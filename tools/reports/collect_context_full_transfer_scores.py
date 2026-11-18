"""Score three frozen source roles only after full runtime/HF/output admission.

No inference dispatch or policy selection. Duplicate policy roles must reuse
the exact same executed operations. Target labels are opened last.
"""
import argparse
import hashlib
import json
from pathlib import Path

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_full_transfer_operations import audit_collected, local_config
from tools.reports.context_transfer_portable_inputs import load as load_portable
from tools.reports.context_transfer_macro_inputs import load as load_original
from tools.reports.evaluate_context_transfer_macro import source_binding
from tools.reports.score_context_transfer_macro import ROLES, score_admitted


def read(path):
    return json.loads(Path(path).read_text())


def match_original_inputs(original,portable):
    """Bind native scorer provenance to the actual portable inference inputs."""
    if original['dataset']!=portable['dataset']:
        raise ValueError('scoring adapter dataset differs from executed dataset')
    if portable['dataset']=='SLURP_clean_TEST60' and original.get('source_release')!=portable['source_release']:
        raise ValueError('original clean TEST ontology differs')
    admitted=load_original(original)
    for key in ('dataset','population','candidate_ids','input_receipts','processor_sha256'):
        if admitted[key]!=portable[key]:raise ValueError('original scoring inputs differ from inference: '+key)
    if [{k:row[k] for k in portable['source_rows'][0]} for row in admitted['source_rows']]!=portable['source_rows']:
        raise ValueError('original rendering differs from portable execution')
    return admitted


def score(c):
    entries=c['entries']
    if set(entries)!=set(ROLES):
        raise ValueError('all three predeclared roles required')
    reports={};audited={};bindings={};common=None;portable=None;proofs={}
    for role in ROLES:
        entry=entries[role];ops=entry['operations'];policy=entry['policy_id']
        package=Path(ops['prepared'])/'package'
        e=read(package/'evaluation.json')
        origin=ops['source_admission']
        # Bind the requested role to the original source curve, not to the
        # path-dependent portable copy of source-evaluation.json.
        requested=dict(origin,policy_role=role,policy_id=policy,dataset=e['dataset'])
        if e['dataset']=='SLURP_clean_TEST60':requested['source_release']=e['source_release']
        binding=source_binding(requested)
        if binding['policy_id']!=policy or e['policy_id']!=policy:
            raise ValueError('requested role and executed policy differ')
        identity=dict(dataset=e['dataset'],source=origin['source_evaluation'],
            curve=origin['source_curve_config'],runtime=e['macro_client']['runtime_identity'],
            inputs={k:e['portable_full_inputs'][k] for k in ('manifest_sha256','population_evidence_sha256')},
            source_release=e.get('source_release'))
        if common is None:common=identity
        if identity!=common:
            raise ValueError('same original source, full inputs, runtime and ontology required')
        bindings[role]=binding
        if policy in audited:
            if ops!=audited[policy]['operations']:
                raise ValueError('identical policy must reuse exact execution and collection')
            reports[role]=audited[policy]['report'];proofs[role]=audited[policy]['proof']
            continue
        proof=audit_collected(ops)
        if (proof['status']!='full_portable_transfer_outputs_audited'
                or proof['runtime_provenance_verified'] is not True
                or proof['parity']['exact'] is not True or proof['policy_id']!=policy
                or proof['dataset']!=e['dataset'] or proof['policy_role']!=e['policy_role']
                or proof['files_sha256']!=ops['manifest_sha256']):
            raise ValueError('actual full runtime, source and HF parity admission required')
        full=load_portable(local_config(package,e)['portable_full_inputs'])
        if portable is None:portable=full
        for key in ('dataset','population','candidate_ids','input_receipts','processor_sha256','source_rows'):
            if full[key]!=portable[key]:raise ValueError('executed role inputs differ: '+key)
        predictions=Path(ops['operations_root'])/'collected/evaluation/inference/predictions.json'
        raw=predictions.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=proof['inference_audit']['predictions_sha256']:
            raise ValueError('predictions changed after full runtime audit')
        inference=proof['inference_audit']
        if inference['policy_id']!=policy:raise ValueError('inference policy differs')
        report=dict(proof=inference,records=json.loads(raw)['records'])
        audited[policy]=dict(operations=ops,report=report,proof=proof)
        reports[role]=report;proofs[role]=proof
    # The original adapters preserve speaker/language/source provenance needed
    # by native scorers; portable inference intentionally omits those fields.
    original=c['original_evaluation']
    if original['dataset']!=common['dataset']:
        raise ValueError('scoring adapter dataset differs from executed dataset')
    if common['dataset']=='SLURP_clean_TEST60' and original.get('source_release')!=common['source_release']:
        raise ValueError('original clean TEST ontology differs')
    admitted=match_original_inputs(original,portable)
    result=score_admitted(reports,admitted,original,c['scoring'])
    result.update(schema='jev-full-transfer-three-role-scores-v1',runtime_provenance_verified=True,
        full_runtime_proofs=proofs,requested_source_bindings=bindings,matched_identity=common,
        scope='Full native metrics after original source, actual container/transport, full input/output '
              'and same-policy HF panel audits. No target tuning or semantic-presence claim.')
    return result


def collect(c,provenance):
    output=Path(c['output']);check_paths({'full_transfer_scores':output})
    if output.exists():raise FileExistsError(output)
    result=score(c);result['config_provenance']=provenance
    output.mkdir(parents=True,exist_ok=False)
    commit_json(output/'config.json',c)
    commit_json(output/'scores.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    result=collect(config,provenance)
    print(json.dumps(dict(dataset=result['dataset'],unique_policies=result['unique_policies'],
                         comparisons=result['comparisons']),indent=2))
