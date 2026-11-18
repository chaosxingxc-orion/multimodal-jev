"""Score the predeclared population only after independent inference audit."""
import argparse
import json
from pathlib import Path

from multimodal_jev.context_event_metrics import ContextEventMetrics
from tools.jev.context_rl_evaluation_plan import sha
from tools.reports.collect_minds_context_pair import audit
from tools.reports.minds_context_runtime import checked_prediction


FIELDS=('query_macro_set_f1','exact_set','top1_event_accuracy','top1_macro_f1',
        'candidate_binary_nll','candidate_brier','empty_selection_rate','multiple_selection_rate')


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def artifact(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def bind(protocol_path,input_root,output):
    protocol=read(protocol_path);root=Path(input_root);m=read(root/'MANIFEST.json')
    inventory=protocol['population']['inventory']
    # Hash only here; no target record parsing before accepted model inference.
    if (sha(inventory['path'])!=inventory['sha256'] or m['recordings']!=8168
            or m['protocol']['sha256']!=sha(protocol_path)):
        raise ValueError('Frozen protocol and complete target population required')
    result=dict(schema='minds-native-descriptor-scoring-v1',threshold=.5,target_tuning=False,
        artifacts=dict(protocol=artifact(protocol_path),inventory=inventory,
            **{name.replace('.','_'):artifact(root/name) for name in
               ('MANIFEST.json','inputs.jsonl','definitions.json','conversion.jsonl')}))
    with Path(output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result


def mean_metrics(groups):
    """Equal group weights; do not turn an infinite NLL sentinel into zero."""
    return {k:None if any(g[k] is None for g in groups) else sum(g[k] for g in groups)/len(groups)
            for k in FIELDS}


def score_records(predictions,inputs,receipts,truth,candidates,policy_id,*,request_identity_protocol='legacy_content_hash'):
    if request_identity_protocol not in ('legacy_content_hash','macro_compact_json_v1'):
        raise ValueError('explicit registered request identity protocol required')
    ids=[r['id'] for r in inputs]
    if (not ids or len(set(ids))!=len(ids) or len(predictions)!=len(ids) or len(receipts)!=len(ids)
            or set(truth)!=set(ids) or len(candidates)!=14 or len(set(candidates))!=14):
        raise ValueError('Complete aligned population and native14 candidates required')
    groups={};details=[]
    def add(key,uid,gold,prob):
        if key not in groups:groups[key]=dict(metric=ContextEventMetrics(candidates,threshold=.5),empty=0,multiple=0,n=0)
        g=groups[key];g['metric'].add(uid,gold,prob);n=sum(p>=.5 for p in prob)
        g['empty']+=n==0;g['multiple']+=n>1;g['n']+=1
    for r,receipt,p in zip(inputs,receipts,predictions,strict=True):
        uid=r['id'];label=truth[uid];gold=label['intent'];variant=label['language']
        if p['id']!=uid or receipt['id']!=uid or receipt['audio_sha256']!=r['audio_sha256']:
            raise ValueError('Prediction/target audio order differs')
        if request_identity_protocol=='legacy_content_hash':
            checked_prediction(r,receipt,p,policy_id)
        else:
            from multimodal_jev.context_macro_rollout import digest
            from multimodal_jev.context_logits_wire import validate_logits_record
            identity=dict(policy_id=policy_id,positions=receipt['positions'][0],
                inputs={k:v for k,v in receipt['tensors'].items() if k!='positions'})
            validate_logits_record(p)
            if (p['policy_id']!=policy_id or p['request_identity']!=identity
                    or p['request_id']!=digest(identity) or len(p['prob'])!=14):
                raise ValueError('macro engine policy/input/probability differs')
        # English variants are pooled by recording, not averaged as three languages.
        language=variant.split('-')[0]
        for key in ('all','variant:'+variant,'language:'+language):add(key,uid,gold,p['prob'])
        chosen={k for k,v in zip(candidates,p['prob'],strict=True) if v>=.5}
        top=candidates[max(range(14),key=lambda i:p['prob'][i])]
        details.append(dict(id=uid,language=variant,gold=gold,top1_correct=top==gold,
            set_f1=2*int(gold in chosen)/(len(chosen)+1),selected_count=len(chosen)))
    reports={}
    for key,g in groups.items():
        reports[key]=dict(g['metric'].result(),empty_selection_rate=g['empty']/g['n'],
            multiple_selection_rate=g['multiple']/g['n'])
    variants={k.removeprefix('variant:'):v for k,v in reports.items() if k.startswith('variant:')}
    languages={k.removeprefix('language:'):v for k,v in reports.items() if k.startswith('language:')}
    return dict(policy_id=policy_id,recording_weighted=reports['all'],variants=variants,languages=languages,
        variety_equal_mean=mean_metrics(list(variants.values())),
        language_equal_mean=mean_metrics(list(languages.values())),records=details)


def compare(initial,selected):
    a,b=initial['records'],selected['records']
    if ([{k:r[k] for k in ('id','language','gold')} for r in a]!=
            [{k:r[k] for k in ('id','language','gold')} for r in b]):
        raise ValueError('Same paired target population required')
    def delta(x,y):return {k:None if x[k] is None or y[k] is None else y[k]-x[k] for k in FIELDS}
    return dict(recording_weighted_delta=delta(initial['recording_weighted'],selected['recording_weighted']),
        variety_equal_mean_delta=delta(initial['variety_equal_mean'],selected['variety_equal_mean']),
        language_equal_mean_delta=delta(initial['language_equal_mean'],selected['language_equal_mean']),
        variant_delta={k:delta(initial['variants'][k],selected['variants'][k]) for k in initial['variants']},
        top1_improved=sum(not x['top1_correct'] and y['top1_correct'] for x,y in zip(a,b,strict=True)),
        top1_regressed=sum(x['top1_correct'] and not y['top1_correct'] for x,y in zip(a,b,strict=True)),
        scope='Source-selected descriptive paired transfer; no target selection or speaker/seed-level significance.')


def run(spec,output):
    if Path(output).exists():raise ValueError('New score output required')
    entry=spec['scoring_contract']
    if sha(entry['path'])!=entry['sha256']:raise ValueError('Scoring contract changed')
    contract=read(entry['path'])
    if (contract['schema']!='minds-native-descriptor-scoring-v1' or contract['threshold']!=.5
            or contract['target_tuning'] is not False):raise ValueError('Fixed scoring contract required')
    args={k:spec[k] for k in ('package','digest','collection')}
    for key in ('launch','observed','transport'):args[key]=read(spec[key])
    inference=audit(**args)
    prior=read(spec['inference_report'])
    if (any(prior.get(k)!=v for k,v in inference.items())
            or prior['runtime_inspect']!=artifact(spec['observed'])):
        raise ValueError('Actual independent inference proof differs')
    artifacts=contract['artifacts'];package=Path(spec['package'])
    for item in artifacts.values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen scoring input changed')
    for name in ('MANIFEST.json','inputs.jsonl','definitions.json','conversion.jsonl'):
        if sha(package/'inputs'/name)!=artifacts[name.replace('.','_')]['sha256']:
            raise ValueError('Scoring population differs from executed inputs')
    if sha(package/'processor/protocol.json')!=artifacts['protocol']['sha256']:
        raise ValueError('Executed protocol differs')
    # No target inventory parsing before independent actual runtime audit above.
    inventory=read(artifacts['inventory']['path']);protocol=read(artifacts['protocol']['path'])
    original={r['id']:r for r in inventory['records']}
    inputs=rows(package/'inputs/inputs.jsonl');ledger=rows(package/'inputs/conversion.jsonl')
    receipts=rows(package/'processor/inputs.jsonl')
    candidates=[c['id'] for c in protocol['candidates']]
    if (len(original)!=len(inventory['records']) or len(original)!=8168 or len(inputs)!=8168
            or set(original)!={r['id'] for r in inputs} or inventory['intent_ids']!=candidates):
        raise ValueError('Complete frozen native population and ontology required')
    truth={}
    for r,e in zip(inputs,ledger,strict=True):
        o=original[r['id']]
        if (e['id']!=r['id'] or e['source_audio_sha256']!=o['encoded_audio_sha256']
                or e['target_audio_sha256']!=r['audio_sha256'] or e['source_samples']!=o['samples']
                or e['target_samples']!=2*o['samples'] or o['upstream_split']!='train'):
            raise ValueError('Gold does not match original/derived audio identity')
        truth[r['id']]=dict(intent=o['intent'],language=o['language'])
    source=read(package/'frozen-source/PACKAGE.json');policies={}
    for role in ('initial','selected'):
        prediction=inference['predictions'][role]
        if sha(prediction['path'])!=prediction['sha256']:raise ValueError('Predictions changed after audit')
        policies[role]=score_records(read(prediction['path'])['records'],inputs,receipts,truth,candidates,
                                    source['policies'][role]['policy_id'])
        if len(policies[role]['variants'])!=14 or len(policies[role]['languages'])!=12:
            raise ValueError('Full 14-variety and 12-language population required')
    result=dict(recordings=8168,decisions=14,policies=policies,
        comparison=compare(policies['initial'],policies['selected']),scoring_contract=entry,
        inference_proof=artifact(spec['inference_report']),official_test=False,target_tuning=False,
        scope='Full local upstream TRAIN frozen transfer diagnostic, native labels; not paper split reproduction or semantic-definition certification.')
    with Path(output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(read(a.spec),a.output)
    print(json.dumps(dict(recordings=r['recordings'],comparison=r['comparison']),indent=2))
