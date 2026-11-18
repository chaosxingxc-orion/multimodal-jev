#!/usr/bin/env python3
"""Score target predictions only after source-only checkpoint choice is bound."""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.fsc_transfer_population import scoring_targets
from context_cache_receipts import parse_stream
from fsc_context_metrics import score,compare
from prepare_fsc_context_transfer import compile_native
from multimodal_jev.slurp_training import content_hash


def run(manifest_path):
    config=json.loads(Path(manifest_path).read_text())
    def read(entry):
        if file_hash(entry['path'])!=entry['sha256']:raise ValueError('bound artifact changed')
        return json.loads(Path(entry['path']).read_text())
    selection=read(config['source_selection'])
    if not selection['complete'] or selection['selected_checkpoint'] is None:
        raise ValueError('complete frozen source selection required before target labels')
    policies=config['policies']
    if set(policies)!={'initial','selected'}:raise ValueError('exact preselected policy pair required')
    if (policies['initial']['policy_id']!=selection['initial_policy_id'] or
            policies['selected']['policy_id']!=selection['selected_checkpoint']['policy_id']):
        raise ValueError('target policies not selected solely by source curve')
    population=read(config['population']);task=read(config['task'])
    proof=read(config['input_audit'])
    stream=config['input_receipts']
    if (file_hash(stream)!=proof['independent_receipts_sha256'] or proof['state']['ExitCode']!=0 or
            proof['state']['OOMKilled'] or proof['manifest']['recordings']!=3793):
        raise ValueError('complete independently checked input stream required')
    receipt_rows,_=parse_stream(Path(stream).read_text());receipts=[r['value'] for r in receipt_rows[1:-1]]
    if population['split']!='test' or population['recordings']!=3793:raise ValueError('full official target test required')
    package=read(config['input_package_manifest'])
    if (package['task_sha256']!=config['task']['sha256'] or
            package['population_sha256']!=config['population']['sha256'] or
            proof['manifest']['request_manifest_sha256']!=config['input_package_manifest']['sha256']):
        raise ValueError('input definitions or population differ from actual processor audit')
    if len(receipts)!=len(population['records']):raise ValueError('input population incomplete')
    for row,receipt in zip(population['records'],receipts,strict=True):
        request=compile_native(row['audio'],task)
        if row['id']!=receipt['id'] or content_hash(request.model_input())!=receipt['request_sha256']:
            raise ValueError('actual request differs from scoring ontology')
    # No target-label lookup until all preselection identities above pass.
    native=scoring_targets(population,task)
    lookup={tuple(c['values'][f] for f in ('action','object','location')):c['id'] for c in task['commands']}
    targets={r['id']:lookup[tuple(r['gold'][f] for f in ('action','object','location'))] for r in native}
    ids=[c['id'] for c in task['commands']];reports={}
    for name,entry in policies.items():
        predictions=read(entry['predictions'])['records']
        reports[name]=score(predictions,population['records'],targets,ids,
                            policy_id=entry['policy_id'],input_receipts=receipts)
    return dict(policies=reports,comparison=compare(reports['initial'],reports['selected']),
        manifest_sha256=file_hash(manifest_path),runtime_provenance_verified=False,
        scope='Complete target scoring; runtime/weight provenance requires independent collectors. No target tuning.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=run(a.manifest)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
