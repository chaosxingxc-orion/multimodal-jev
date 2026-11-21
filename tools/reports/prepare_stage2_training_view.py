"""Publish/audit a TRAIN-only difficulty view from a completed Stage1 reference.

The producer audit is replayed using the existing reference journal auditor.
No model call, dataset copying, automatic training or test feedback.
"""
import argparse,json
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import ReferenceDataset
from multimodal_jev.slurp_portable_release import load
from multimodal_jev.stage2_training_view import publish,sample,load as load_view


def inputs(c,provenance):
    from tools.reports.prepare_context_reference_job import audit_results
    if c['schema']!='jev-stage2-view-preparation-v1':raise ValueError('explicit view preparation required')
    proof=audit_results(c['reference_job'],provenance)['output_audit']
    job=c['reference_job'];result=json.loads((Path(job['collection'])/'reference-build/result.json').read_text())
    build=json.loads((Path(job['output'])/'reference-build-config.json').read_text())
    rows,task,_=load(build['release'],build['release_sha256'],'learning',purpose='optimize')
    if (len(rows)!=build['expected_population'] or proof['recordings']!=len(rows)
            or proof['policy_id']!=result['identity']['policy_id'] or proof['manifest_sha256']!=result['manifest_sha256']
            or result['identity']['release_sha256']!=build['release_sha256']):raise ValueError('audited full TRAIN endpoint scores required')
    binding=dict(root=str(Path(job['collection'])/'reference-build/dataset'),manifest_sha256=result['manifest_sha256'],identity=result['identity'],population=result['population'])
    reference=ReferenceDataset(binding['root'],manifest_sha256=binding['manifest_sha256'],expected_identity=binding['identity'],expected_population=binding['population'])
    return rows,task,reference,binding,proof


def run(c,p,*,audit=False):
    rows,task,reference,binding,proof=inputs(c,p)
    if not audit:
        result=publish(c['output'],rows=rows,task=task,reference=reference,reference_binding=binding,options=c['sampling'],batch_config=c['batch_config'])
        commit_json(Path(c['output'])/'producer-audit.json',proof)
        commit_json(Path(c['output'])/'preparation.json',dict(result=result,provenance=p))
        return result
    root=Path(c['output']);m=json.loads((root/'MANIFEST.json').read_text())
    if m['options']!=c['sampling']:raise ValueError('sampler options differ')
    expected_config=dict(c['batch_config'],source_queries=m['batch_config']['source_queries'])
    chosen,batches,_=load_view(root,c['manifest_sha256'],rows=rows,task=task,reference_binding=binding,batch_config=expected_config)
    expected=sample(rows,binding['identity']['candidate_ids'],{r['id']:reference.read_sample(r['id'])['prob'] for r in rows},c['sampling'])
    if json.loads((root/'selection.json').read_text())!=dict(selected_ids=expected[0],diagnostics=expected[1],summary=expected[2]):raise ValueError('difficulty selection not independently reproducible')
    result=dict(status='stage2_training_view_independently_reproduced',manifest_sha256=file_hash(root/'MANIFEST.json'),recordings=len(chosen),updates=len(batches),reference_manifest_sha256=reference.manifest_sha256,npu_admitted=False,training_dispatch_allowed=False)
    commit_json(c['audit_output'],result);return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--local',required=True);a.add_argument('--action',choices=['prepare','audit'],default='prepare');x=a.parse_args();c,p=load_config(x.config,x.local);print(json.dumps(run(c,p,audit=x.action=='audit'),indent=2))
