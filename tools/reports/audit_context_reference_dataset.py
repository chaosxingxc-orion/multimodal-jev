"""Audit a built TRAIN reference dataset against independent input receipts.

This is artifact/journal verification only. Actual services, container scope and
actor/vLLM parity still need their separately bound admission evidence.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_audit import audit
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.slurp_portable_release import load
from tools.reports.build_context_reference_dataset import validate_endpoint
from multimodal_jev.context_policy_artifacts import read_execution_bundle


def run(c,provenance):
    for key in ('build_config','stage1_commit','input_receipts'):
        if file_hash(c[key])!=c[key+'_sha256']:raise ValueError('bound reference evidence changed: '+key)
    build=json.loads(Path(c['build_config']).read_text())
    root=Path(c['build_output']);result=json.loads((root/'result.json').read_text())
    if json.loads((root/'terminal.json').read_text())!={'exit_code':0}:
        raise ValueError('complete successful reference build required')
    bundle=read_execution_bundle(c['stage1_policy'],expected_policy_id=build['source_policy_id'],
        expected_contract=build['source_contract'],expected_base_id=build['base_id'])
    checkpoint=json.loads(Path(c['stage1_commit']).read_text())
    validate_endpoint(checkpoint,build,bundle)
    if checkpoint['joint_policy_manifest_sha256']!=file_hash(Path(c['stage1_policy'])/'MANIFEST.json'):
        raise ValueError('actual reference source differs from committed Stage1 snapshot')
    if build['pins'].get(build['stage1_commit'])!=c['stage1_commit_sha256']:
        raise ValueError('build endpoint differs from audited endpoint')
    rows,task,_=load(build['release'],build['release_sha256'],'learning',purpose='optimize')
    candidates=[dict(id=v['id'],definition=v['description']) for v in task['schema']['candidates']]
    from tools.reports.context_reference_input_evidence import load as load_inputs,bindings_from_records
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    evidence,input_audit=load_inputs(c,rows=rows,task=task,release_sha256=build['release_sha256'],
        processor_sha256=processor_cache_identity(c['stage1_policy']))
    bindings=bindings_from_records(evidence,json.loads((root/'dataset/records.json').read_text()),
        result['population'],policy_id=bundle['policy_id'])
    population={k:digest(v) for k,v in bindings.items()}
    identity=dict(schema='jev-reference-dataset-v1',policy_id=bundle['policy_id'],base_id=build['base_id'],
        contract=build['source_contract'],runtime_identity=build['macro_client']['runtime_identity'],
        release_sha256=build['release_sha256'],candidate_ids=[v['id'] for v in candidates],population_sha256=digest(population))
    if (result['identity']!=identity or result['population']!=population
            or result['recordings']!=len(rows)):
        raise ValueError('reference build population/identity differs')
    proof=audit(root/'dataset',c['journal'],manifest_sha256=result['manifest_sha256'],identity=identity,
        population=population,input_bindings=bindings,macro_queries=build['macro_client']['max_queries'],
        rollout_queries=build['macro_client']['rollout_queries'],replicas=len(build['macro_client']['replicas']))
    proof.update(input_receipts_sha256=c['input_receipts_sha256'],stage1_commit_sha256=c['stage1_commit_sha256'],
                 input_proof_sha256=c['input_proof_sha256'],input_audit=input_audit,
                 build_config_sha256=c['build_config_sha256'],provenance=provenance)
    commit_json(c['output'],proof);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance),indent=2))
