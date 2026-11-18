#!/usr/bin/env python3
"""Real CPU processor budget for input-only routing contract candidates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from audit_harness_routing_support import policy, FAMILIES
from multimodal_jev.baseline import file_hash, atomic_json
from multimodal_jev.span_contract import SpanQuestion, SpanRequest, encode_span_request, span_labels
from multimodal_jev.task_schema import Candidate, DecisionSchema
from multimodal_jev.slurp_training import content_hash


def request_for(audio, task, caller_policy, *, structured, representation='literal_v1'):
    """No labels enter template compilation; explicit event ontology is retained."""
    if representation not in ('literal_v1', 'compact_json_v1'):
        raise ValueError('registered representation required')
    schema=DecisionSchema.from_dict(task['schema'])
    ids={c.id for c in schema.candidates}
    routes=caller_policy['routes']
    if (type(structured) is not bool or len({r['id'] for r in routes})!=len(routes)
            or caller_policy['fallback']!='no_route' or not routes
            or any(r['id']=='no_route' or type(r['enabled']) is not bool
                   or not r['accepts'] or not set(r['accepts'])<=ids for r in routes)):
        raise ValueError('explicit unique routes, known events and availability required')
    requirements=tuple(task['thinking']['requirements'])+(
        'Interpret the audio using the supplied exclusive event ontology and its annotation policy. '
        'A route is applicable exactly when enabled and its accepts list contains that event. '
        'Several routes may be applicable. Select any applicable route; select no_route only if none applies. '
        'no_route does not mean unintelligible audio or failed execution.',
        json.dumps(dict(event_ontology=asdict(schema),caller_routing_policy=caller_policy),sort_keys=True,
                   **(dict(separators=(',', ':')) if representation=='compact_json_v1' else {})),)
    questions=[]
    if structured:
        questions=[SpanQuestion('applicable-'+r['id'],DecisionSchema(
            id='applicable-'+r['id'],decision_type='noul',allow_unknown=False,
            question='Under the supplied routing policy, is '+r['id']+' applicable to this spoken request?')) for r in routes]
    candidates=tuple(Candidate(r['id'],'Select route '+r['id']+' under its supplied acceptance and availability rule.') for r in routes)
    candidates+=(Candidate('no_route','No enabled route accepts the event under the supplied routing policy.'),)
    questions.append(SpanQuestion('final',DecisionSchema('harness-route-final-v1','choice',
        'Which supplied route can receive this request under the routing policy?',candidates,False)))
    return SpanRequest(audio,tuple(questions),'final',requirements,'harness-routing-context-candidate-v1')


def run(config, output):
    import inspect
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only probe required')
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('pinned input changed: '+p)
    output=Path(output);output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    task=json.loads(Path(config['tasks']).read_text())['intent']
    records=[]
    for asset in config['audio']:
        if file_hash(asset['audio'])!=asset['sha256']:raise ValueError('audio changed')
        wave,receipt=load_audio(asset['audio'],return_receipt=True)
        for family in FAMILIES:
            spec=policy(sorted(task['candidate_pairs']),family,317)
            for structured in (False,True):
                representation=config.get('representation','literal_v1')
                request=request_for(asset['audio'],task,spec,structured=structured,representation=representation)
                original=request_for(asset['audio'],task,spec,structured=structured)
                # Compare the full input object with the JSON policy decoded;
                # only insignificant JSON whitespace may change.
                def semantic_input(value):
                    wire=value.model_input()
                    wire['requirements']=list(wire['requirements'])
                    wire['requirements'][-1]=json.loads(wire['requirements'][-1])
                    return wire
                if semantic_input(request)!=semantic_input(original):
                    raise ValueError('representation changed input contract')
                inputs,positions=encode_span_request(processor,request,wave)
                mappings=span_labels(processor.tokenizer,request.questions)
                length=int(inputs['attention_mask'].sum())
                records.append(dict(audio_id=asset['id'],audio_sha256=asset['sha256'],audio_receipt=receipt,
                    family=family,structured=structured,representation=representation,
                    decoded_contract_sha256=content_hash(semantic_input(request)),
                    policy_sha256=content_hash(spec),
                    request_sha256=content_hash(request.model_input()),tokens=length,
                    span_count=len(request.questions),positions=positions.tolist(),mappings=mappings,
                    input_ids_sha256=content_hash(inputs['input_ids'].tolist()),
                    overflow={str(n):length>n for n in (4096,8192,16384)}))
                atomic_json(output/'progress.json',dict(completed=len(records)))
    expected=len(config['audio'])*len(FAMILIES)*2
    if len(records)!=expected:raise ValueError('incomplete context cases')
    atomic_json(output/'records.json',records)
    report=dict(requests=len(records),audio_assets=len(config['audio']),records_sha256=file_hash(output/'records.json'),
        config_sha256=content_hash(config),processor_source_sha256=file_hash(inspect.getfile(type(processor))),
        minimum_tokens=min(r['tokens'] for r in records),maximum_tokens=max(r['tokens'] for r in records),
        overflow={str(n):sum(r['tokens']>n for r in records) for n in (4096,8192,16384)},
        context_accepted=False,training_dispatch_allowed=False,
        scope='Actual processor/template and Span alignment on median/longest learning audio for four fixed caller policies. No model forward, all-policy/full-population capacity, NPU, semantics or training admission.')
    atomic_json(output/'result.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();print(json.dumps(run(json.loads(a.config.read_text()),a.output)),flush=True)
