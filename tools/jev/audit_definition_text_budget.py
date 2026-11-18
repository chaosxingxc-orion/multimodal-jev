#!/usr/bin/env python3
"""Measure supplied task text with the pinned tokenizer; no audio/model execution."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import string

from multimodal_jev.span_contract import SpanQuestion, SpanRequest, span_labels
from multimodal_jev.task_schema import Candidate, DecisionSchema


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda path:json.loads(path.read_text())
    files={k:a.root/'configs/tasks/slurp'/v for k,v in dict(
        intent='intent-definition-candidates-v1.json',slot='type-definition-candidates-v1.json',
        rules='type-boundary-rules-candidate-v2.json').items()}
    intent=read(files['intent'])['cards'];slots=read(files['slot'])['types'];rules=read(files['rules'])['rules']
    symbols=string.ascii_uppercase+string.ascii_lowercase+string.digits
    imap={c['semantic_id']:symbols[i] for i,c in enumerate(intent)}
    smap={c['entity_type']:'S'+str(i+1) for i,c in enumerate(slots)}
    # IDs and evidence remain outside prompt, while natural-language definitions remain intact.
    rename=lambda text:__import__('re').sub('|'.join(sorted(map(__import__('re').escape,imap.keys()|smap.keys()),key=len,reverse=True)),
        lambda m:imap.get(m.group(),smap.get(m.group())),text)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    count=lambda text:len(tokenizer.encode(text,add_special_tokens=False))
    core_rules=('Use only the audio and supplied definitions. Return each typed decision once; do not generate reasoning.',
        'Intent membership follows the exclusive task policy. A slot requires an expressed role satisfying its definition, not merely a parameter usually required by the intent.',
        'No slot value extraction or external composition. Final is the full intent choice. Multiple slot types may coexist.')
    cases=[]
    for description_mode in ('definition_only','definition_and_boundaries'):
        candidates=[]
        for c in intent:
            description=c['definition']
            if description_mode=='definition_and_boundaries':description+=' Includes: '+' '.join(c['includes'])+' Excludes: '+' '.join(c['excludes'])
            candidates.append(Candidate(imap[c['semantic_id']],rename(description)))
        # Every referenced definition must precede every readout under causal attention.
        final_candidates=tuple(Candidate(c.id,'Use supplied definition '+c.id+'.') for c in candidates)
        final=SpanQuestion('Final',DecisionSchema('Final','choice','Which complete defined intent class best describes the spoken request?',final_candidates,False))
        for knowledge in ('intent_only','all_slots','all_slots_and_boundaries'):
            reqs=list(core_rules)
            reqs.append('Intent definitions: '+json.dumps({c.id:c.description for c in candidates},ensure_ascii=False))
            if knowledge!='intent_only':
                reqs.append('Slot definitions: '+json.dumps({smap[c['entity_type']]:rename(c['definition']) for c in slots},ensure_ascii=False))
            if knowledge=='all_slots_and_boundaries':
                reqs.append('Shared slot rules: '+json.dumps([dict(types=[smap[t] for t in r['types']],rule=rename(r['shared_rule']),coexistence=rename(r['coexistence_rule'])) for r in rules],ensure_ascii=False))
            for size in (0,2,8):
                # Fixed first definitions only: structural design probe, not full menu audit.
                questions=[]
                for i in range(size//2):
                    questions.append(SpanQuestion('I'+str(i+1),DecisionSchema('I'+str(i+1),'noul',
                        'Does the query belong to class '+imap[intent[i]['semantic_id']]+' under the supplied exclusive intent definition?',allow_unknown=False)))
                    # If global slot definitions absent, include each referenced slot locally.
                    desc=rename(slots[i]['definition']) if knowledge=='intent_only' else 'See supplied slot definition.'
                    questions.append(SpanQuestion('S'+str(i+1),DecisionSchema('S'+str(i+1),'noul',
                        'Does the query express role '+smap[slots[i]['entity_type']]+'? '+desc+' Do not extract its value.',allow_unknown=False)))
                questions.append(final)
                request=SpanRequest('/design-only-no-audio.wav',tuple(questions),'Final',tuple(reqs))
                mappings=span_labels(tokenizer,request.questions)
                system=('English\nFill all typed decision spans from the audio in one forward pass. '
                        'Intermediate and final questions are provided. Do not generate text.\n'
                        +json.dumps({'version':request.version,'requirements':request.requirements},ensure_ascii=False))
                parts=[json.dumps({'id':q.id,'final':q.id==request.final_id,'schema':asdict(q.schema),
                    'symbols':dict(zip(mappings[i]['symbols'],q.values))},ensure_ascii=False)+f'\n[JEV_SPAN_{i+1}]'
                    for i,q in enumerate(request.questions)]
                payload=system+'\n'+'\n'.join(parts)
                # Also measure a proposed dictionary rendering; not implemented in the actor.
                compact=json.dumps(dict(rules=reqs,
                    questions=[dict(id=q.id,question=q.schema.question,answers=['false','true'] if q.schema.decision_type=='noul' else list(imap.values())) for q in request.questions]),ensure_ascii=False,separators=(',',':'))
                cases.append(dict(description_mode=description_mode,knowledge=knowledge,auxiliary_Q=size,
                    existing_schema_payload_tokens=count(payload),compact_dictionary_payload_tokens=count(compact),
                    existing_payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                    compact_payload_sha256=hashlib.sha256(compact.encode()).hexdigest(),
                    existing_text=payload,compact_text=compact))
    a.output.mkdir(parents=True)
    (a.output/'payloads.json').write_text(json.dumps(cases,indent=2)+'\n')
    result=dict(schema='slurp-definition-text-budget-v1',cases=[{k:v for k,v in c.items() if not k.endswith('_text')} for c in cases],
        pins={str(p):sha(p) for p in [*files.values(),a.model/'tokenizer.json',a.model/'tokenizer_config.json',Path(__file__),a.root/'src/multimodal_jev/span_contract.py']},
        payloads_sha256=sha(a.output/'payloads.json'),
        causal_definition_placement='All referenced intent definitions in system requirements before any question; slots global or in their own question.',
        scope='Actual tokenizer over supplied task payload, without chat-template/audio expansion. Not full context length, full menu coverage, implemented compact renderer, or task-semantic acceptance.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['cases'],indent=2))


if __name__=='__main__':main()
