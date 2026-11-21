"""Build/audit/bind a TRAIN-only difficulty release from audited endpoint cache."""
import argparse
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_component_registry import resolve
from multimodal_jev.context_reference_dataset import ReferenceDataset
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev import stage2_dataset_release as dataset


def inputs(c,provenance):
    from tools.reports.prepare_context_reference_job import audit_results
    dataset.validate_execution(c['execution'])
    if c['schema']!='jev-stage2-dataset-preparation-v2':raise ValueError('explicit dataset preparation schema required')
    # This replays the actual producer/input journal audit, not a PASS string.
    result=audit_results(c['reference_job'],provenance);proof=result['output_audit']
    job=c['reference_job'];root=Path(job['collection'])
    build=json.loads((Path(job['output'])/'reference-build-config.json').read_text())
    produced=json.loads((root/'reference-build/result.json').read_text())
    reader=resolve('release','release_handle_v1')()
    rows,task,_,_=reader.load(build['release'],build['release_sha256'],'learning','optimize')
    reference=ReferenceDataset(root/'reference-build/dataset',manifest_sha256=produced['manifest_sha256'],
        expected_identity=produced['identity'],expected_population=produced['population'],execution=c['execution'])
    # The collector transports outputs. The input commit is in the sealed
    # producer package and is independently checked by audit_results.
    commit=Path(job['output'])/'initial/endpoint-commit.json';commit_sha=file_hash(commit)
    endpoint=dataset.endpoint_binding(json.loads(commit.read_text()),commit_sha256=commit_sha,
        reference=reference,source_expected=c['stage1_expected'],sampling=c['sampling'])
    producer=json.loads((root/'reference-build/dataset/MANIFEST.json').read_text())['producer']
    if (producer.get('stage1_commit_sha256')!=commit_sha or producer.get('endpoint')!=endpoint['endpoint']
            or proof['stage1_commit_sha256']!=commit_sha
            or proof['policy_id']!=endpoint['policy_id'] or proof['manifest_sha256']!=reference.manifest_sha256
            or len(rows)!=build['expected_population'] or proof['recordings']!=len(rows)):
        raise ValueError('reference producer, source and completed Stage1 differ')
    return rows,task,reference,endpoint


def save_receipt(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        prior=json.loads(path.read_text())
        # Timings/reuse counters are observations, not semantic evidence.
        stable=lambda x:{k:v for k,v in x.items() if k not in ('timings','build_timings','reused')}
        if stable(prior)!=stable(value):raise ValueError('existing receipt binds another result')
    else:commit_json(path,value)


def preview(c,p):
    """Replay the full reference artifacts without admitting the producer job."""
    from tools.reports.prepare_context_reference_job import collected_audit_config,replay_reference_outputs
    from multimodal_jev.context_topology_batch_plan import derive
    dataset.validate_execution(c['execution'])
    if c['schema']!='jev-stage2-dataset-preview-v1':
        raise ValueError('explicit non-release preview config required')
    job=c['reference_job']
    audit_inputs=collected_audit_config(dict(job,results_audit_output=c['receipt']))
    proof=replay_reference_outputs(audit_inputs,p)
    build=json.loads(Path(audit_inputs['build_config']).read_text())
    root=Path(job['collection'])/'reference-build'
    produced=json.loads((root/'result.json').read_text())
    if (proof['manifest_sha256']!=produced['manifest_sha256']
            or proof['recordings']!=build['expected_population']):
        raise ValueError('preview reference proof differs from full producer result')
    reader=resolve('release','release_handle_v1')()
    rows,task,_,_=reader.load(build['release'],build['release_sha256'],'learning','optimize')
    reference=ReferenceDataset(root/'dataset',manifest_sha256=produced['manifest_sha256'],
        expected_identity=produced['identity'],expected_population=produced['population'],execution=c['execution'])
    endpoint=dataset.endpoint_binding(json.loads(Path(audit_inputs['stage1_commit']).read_text()),
        commit_sha256=audit_inputs['stage1_commit_sha256'],reference=reference,
        source_expected=c['stage1_expected'],sampling=c['sampling'])
    dataset._inputs(rows,task,reference,endpoint,c['sampling'],c['batch_config'],c['evaluation'],c['execution'])
    files=dataset.expected_files(rows,task,reference,c['sampling'],c['batch_config'],c['evaluation'],c['execution'],{})
    summary=files['selection.json']['summary'];consumer=files['consumer.json']
    plan=derive(consumer['batch_config'])
    from tools.reports.context_replica_formal_candidate import validate_schedule
    counts=validate_schedule(files['batches.json'],files['selection.json']['selected_ids'],plan)
    if sum(counts)!=consumer['real_exposures']:
        raise ValueError('preview schedule exposure mismatch')
    result=dict(status='stage2_sampling_preview_only',training_dispatch_allowed=False,
        producer_lifecycle_admitted=False,view_published=False,scope=proof['scope'],
        reference=dict(recordings=proof['recordings'],distinct_inputs=proof['distinct_inputs'],
            macro_batches=proof['macro_batches'],manifest_sha256=proof['manifest_sha256'],
            policy_id=proof['policy_id']),endpoint=endpoint,sampling=c['sampling'],summary=summary,
        consumer=consumer,tail_real_queries=plan['tail_real_queries'],tail_padding_slots=plan['tail_padding_slots'])
    save_receipt(c['receipt'],result)
    return result


def run(c,p,action='prepare'):
    if action=='preview':return preview(c,p)
    if action not in ('prepare','audit','bind','release'):raise ValueError('unknown dataset action')
    receipt=Path(c['receipts'][action])
    paths=[Path(c['output']),receipt]
    if action in ('bind','release'):
        bound=c['stage2_config'];output=Path(c['stage2_output'])
        paths.extend((output,Path(bound['path'])))
    resolved=[path.resolve() for path in paths]
    if any(x.is_relative_to(y) or y.is_relative_to(x) for i,x in enumerate(resolved) for y in resolved[:i]):
        raise ValueError('disjoint data, receipt and configuration paths required')
    if action in ('bind','release'):
        raw=Path(bound['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=bound['sha256']:raise ValueError('input Stage2 configuration changed')
        base=json.loads(raw)
    rows,task,reference,endpoint=inputs(c,p)
    args=dict(rows=rows,task=task,reference=reference,endpoint=endpoint,options=c['sampling'],
        batch_config=c['batch_config'],evaluation=c['evaluation'],execution=c['execution'])
    if action in ('prepare','release'):
        proof=dataset.publish(c['output'],**args)
    else:
        proof=dataset.audit(c['output'],c['manifest_sha256'],**args)
    if action in ('bind','release'):
        config=dataset.bind_config(base,root=c['output'],manifest_sha256=proof['manifest_sha256'],
            consumer=proof['consumer'],reference=reference)
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            if json.loads(output.read_text())!=config:raise ValueError('existing Stage2 config differs')
        else:commit_json(output,config)
        proof=dict(proof,bound_config_sha256=file_hash(output))
    save_receipt(c['receipts'][action],proof)
    return proof


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--config',required=True);a.add_argument('--local',required=True)
    a.add_argument('--action',choices=['prepare','audit','bind','release','preview'],default='prepare');x=a.parse_args()
    c,p=load_config(x.config,x.local);result=run(c,p,x.action)
    print(json.dumps({k:v for k,v in result.items() if k not in ('timings','build_timings')},indent=2))
