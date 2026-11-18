"""Read clean TEST labels only after independent frozen inference verification."""
import argparse
import json
from pathlib import Path

from tools.reports.collect_slurp_context_test_pair import audit
from tools.jev.context_rl_evaluation_plan import sha
from tools.jev.rescore_slurp_clean_native import clean_population
from multimodal_jev.context_event_metrics import ContextEventMetrics
from tools.reports.slurp_context_test_runtime import checked_prediction


def read(p):return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open() as f:return [json.loads(line) for line in f]


def artifact(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))


def bind_contract(input_root, reference_receipt, tasks, output):
    """Bind already published hashes before inference; never parse TEST labels."""
    root=Path(input_root);m=read(root/'MANIFEST.json');receipt=read(reference_receipt)
    result_path=Path(receipt['result'])
    if sha(result_path)!=receipt['result_sha256']:raise ValueError('reference result changed')
    result=read(result_path)
    if (m['recordings']!=12991 or m['training_release_sha256']!=sha(Path(tasks).parent/'MANIFEST.json')
            or m['task_source_sha256']!=sha(tasks) or result['optimization_allowed'] is not False):
        raise ValueError('fixed task and clean TEST required')
    def find(suffix):
        matches=[p for p in result['pins'] if p.endswith(suffix)]
        if len(matches)!=1:raise ValueError('unique pinned source required: '+suffix)
        return dict(path=matches[0],sha256=result['pins'][matches[0]])
    contract=dict(schema='slurp-context-clean-test-scoring-v1',threshold=.5,optimization_allowed=False,
        artifacts=dict(input_manifest=artifact(root/'MANIFEST.json'),definitions=artifact(root/'definitions.json'),
            input_rows=artifact(root/'inputs.jsonl'),tasks=artifact(tasks),reference_receipt=artifact(reference_receipt),
            reference_result=artifact(result_path),population=dict(path=str(result_path.parent/'population.json'),sha256=result['population_sha256']),
            reference_predictions=dict(path=str(result_path.parent/'official-predictions.jsonl'),sha256=result['normalized_predictions_sha256']),
            native_requests=find('/test.requests.jsonl'),native_targets=find('/test.targets.jsonl'),
            clean_metadata=find('/metadata/test.jsonl'),excluded=find('/test.native-excluded-ids.json')))
    with Path(output).open('x') as f:json.dump(contract,f,indent=2);f.write('\n')
    return contract


def reconstruct_targets(requests,targets,metadata,excluded,population,inputs,task):
    kept,clean_targets=clean_population(requests,targets,metadata,excluded)
    ids=[r['id'] for r in inputs]
    if ([r['id'] for r in kept]!=ids or population['ids']!=ids
            or population['excluded_ids']!=excluded):
        raise ValueError('same clean TEST population/order required')
    candidates=[c['id'] for c in task['schema']['candidates']]
    pairs=task['candidate_pairs'];reverse={tuple(v):k for k,v in pairs.items()}
    if set(pairs)!=set(candidates) or len(reverse)!=len(candidates):
        raise ValueError('unambiguous fixed TRAIN candidate pairs required')
    truth={}
    for source,original,label in zip(inputs,kept,clean_targets,strict=True):
        if (source['audio_sha256']!=original['audio_sha256'] or source['audio']!=original['input']['audio']
                or source['source_group']!=original['source_group']):
            raise ValueError('target and executed input audio/group differ')
        native=label['target'];key=(native['scenario'],native['action'])
        if key not in reverse:raise ValueError('TEST gold outside TRAIN ontology')
        truth[source['id']]=reverse[key]
    return truth


def score_records(predictions,inputs,receipts,targets,candidates,policy_id,*,request_identity_protocol='legacy_content_hash'):
    if request_identity_protocol not in ('legacy_content_hash','macro_compact_json_v1'):
        raise ValueError('explicit registered request identity protocol required')
    ids=[r['id'] for r in inputs]
    if (not ids or len(set(ids))!=len(ids) or len(predictions)!=len(ids) or len(receipts)!=len(ids)
            or set(targets)!=set(ids) or len(candidates)!=60 or len(set(candidates))!=60):
        raise ValueError('complete distinct population and 60 candidates required')
    metrics=ContextEventMetrics(candidates,threshold=.5);records=[];empty=multiple=0
    for source,receipt,p in zip(inputs,receipts,predictions,strict=True):
        uid=source['id']
        if (p['id']!=uid or receipt['id']!=uid or receipt['audio_sha256']!=source['audio_sha256']):
            raise ValueError('aligned full audio population required')
        if request_identity_protocol=='legacy_content_hash':
            checked_prediction(source,receipt,p,policy_id)
        else:
            from multimodal_jev.context_macro_rollout import digest
            from multimodal_jev.context_logits_wire import validate_logits_record
            identity=dict(policy_id=policy_id,positions=receipt['positions'][0],
                inputs={k:v for k,v in receipt['tensors'].items() if k!='positions'})
            validate_logits_record(p)
            if (p['policy_id']!=policy_id or p['request_identity']!=identity
                    or p['request_id']!=digest(identity) or len(p['prob'])!=60):
                raise ValueError('macro engine policy/input/probability differs')
        y=targets[uid];metrics.add(uid,y,p['prob'])
        chosen={k for k,v in zip(candidates,p['prob'],strict=True) if v>=.5}
        top=candidates[max(range(60),key=lambda i:p['prob'][i])]
        empty+=not chosen;multiple+=len(chosen)>1
        records.append(dict(id=uid,source_group=source['source_group'],gold=y,top1=top,
            top1_correct=top==y,set_f1=2*int(y in chosen)/(len(chosen)+1),selected_count=len(chosen)))
    return dict(policy_id=policy_id,metrics=metrics.result(),empty_selection_rate=empty/len(ids),
        multiple_selection_rate=multiple/len(ids),records=records)


def compare(initial,selected):
    a,b=initial['records'],selected['records']
    if ([{k:r[k] for k in ('id','source_group','gold')} for r in a]!=
            [{k:r[k] for k in ('id','source_group','gold')} for r in b]):
        raise ValueError('same paired TEST population required')
    fields=('query_macro_set_f1','exact_set','top1_event_accuracy','top1_macro_f1','candidate_binary_nll','candidate_brier')
    return dict(initial_policy_id=initial['policy_id'],selected_policy_id=selected['policy_id'],
        delta={k:None if initial['metrics'][k] is None or selected['metrics'][k] is None else
            selected['metrics'][k]-initial['metrics'][k] for k in fields},
        top1_improved=sum(not x['top1_correct'] and y['top1_correct'] for x,y in zip(a,b)),
        top1_regressed=sum(x['top1_correct'] and not y['top1_correct'] for x,y in zip(a,b)),
        scope='Source-selected policies, descriptive paired TEST difference; no target selection or seed-level significance')


def run(spec,output):
    output=Path(output)
    if output.exists():raise ValueError('new score output required')
    entry=spec['scoring_contract']
    if sha(entry['path'])!=entry['sha256']:raise ValueError('predeclared scoring contract changed')
    contract=read(entry['path'])
    if (contract['schema']!='slurp-context-clean-test-scoring-v1' or contract['threshold']!=.5
            or contract['optimization_allowed'] is not False):raise ValueError('fixed TEST-only scoring required')
    args={k:spec[k] for k in ('package','digest','collection')}
    for name in ('launch','observed','transport'):args[name]=read(spec[name])
    inference=audit(**args)
    prior=read(spec['inference_report'])
    if (any(prior.get(k)!=v for k,v in inference.items())
            or prior['runtime_inspect']!=artifact(spec['observed'])):
        raise ValueError('independent inference/inspection proof differs')
    root=Path(spec['package']);a=contract['artifacts']
    for key,name in [('input_manifest','MANIFEST.json'),('definitions','definitions.json'),('input_rows','inputs.jsonl')]:
        if a[key]['sha256']!=sha(root/'inputs'/name):raise ValueError('scoring inputs differ from execution')
    # Only after actual inference is re-audited do target files enter scoring.
    for item in a.values():
        if sha(item['path'])!=item['sha256']:raise ValueError('frozen scoring source changed')
    m=read(a['input_manifest']['path']);task=read(a['tasks']['path'])['intent']
    if sha(a['tasks']['path'])!=m['task_source_sha256']:raise ValueError('TRAIN ontology changed')
    for key in ('native_requests','native_targets','clean_metadata','excluded','population','reference_predictions','reference_result'):
        item=a[key]
        if m['source_files'].get(item['path'])!=item['sha256']:
            raise ValueError('scoring source not bound to accepted clean input publication: '+key)
    reference_receipt=read(a['reference_receipt']['path'])
    if (reference_receipt['result']!=a['reference_result']['path']
            or reference_receipt['result_sha256']!=a['reference_result']['sha256']):
        raise ValueError('reference receipt/result differ')
    inputs=rows(a['input_rows']['path']);receipts=rows(root/'processor/inputs.jsonl')
    if len(inputs)!=12991:raise ValueError('full clean TEST required')
    definitions=read(a['definitions']['path']);candidates=[c['id'] for c in task['schema']['candidates']]
    if definitions['candidates']!=[dict(id=c['id'],definition=c['description']) for c in task['schema']['candidates']]:
        raise ValueError('executed candidate definitions differ')
    truth=reconstruct_targets(rows(a['native_requests']['path']),rows(a['native_targets']['path']),
        rows(a['clean_metadata']['path']),read(a['excluded']['path']),read(a['population']['path']),inputs,task)
    source=read(root/'frozen-source/PACKAGE.json');policies={}
    for role in ('initial','selected'):
        p=inference['predictions'][role]
        if sha(p['path'])!=p['sha256']:raise ValueError('predictions changed after audit')
        policies[role]=score_records(read(p['path'])['records'],inputs,receipts,truth,candidates,source['policies'][role]['policy_id'])
    ref=rows(a['reference_predictions']['path']);indexed={'slurp:'+r['file']:r for r in ref}
    if len(indexed)!=len(ref) or set(indexed)!=set(truth):raise ValueError('complete native reference required')
    pairs=task['candidate_pairs']
    correct=sum([indexed[uid]['scenario'],indexed[uid]['action']]==pairs[y] for uid,y in truth.items())
    archived=read(a['reference_receipt']['path'])
    if correct!=archived['joint_correct'] or len(inputs)!=archived['examples']:
        raise ValueError('reference joint accuracy does not reproduce')
    result=dict(runtime_provenance_verified=True,recordings=len(inputs),source_groups=len({r['source_group'] for r in inputs}),
        policies=policies,comparison=compare(policies['initial'],policies['selected']),scoring_contract=entry,
        inference_proof=artifact(spec['inference_report']),
        native_omni_reference=dict(correct=correct,accuracy=correct/len(inputs),malformed=archived['malformed'],
            comparison={r:policies[r]['metrics']['top1_event_accuracy']-correct/len(inputs) for r in policies},
            scope='Same clean audio population, different task interface and training. Omni native composite generation counts malformed wrong; no matched-budget or algorithm-superiority claim.'),
        scope='Complete fixed clean TEST subset, exclusive benchmark event only; no slot extraction, semantic-presence truth, target tuning, or pure-RL causal attribution.')
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(read(a.spec),a.output)
    print(json.dumps(dict(recordings=r['recordings'],comparison=r['comparison']),indent=2))
