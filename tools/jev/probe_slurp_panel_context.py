#!/usr/bin/env python3
"""Task-design probe using the unchanged real Span encoder; no model training."""
from dataclasses import asdict
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import string

from multimodal_jev.span_contract import SpanRequest, SpanQuestion, encode_span_request
from multimodal_jev.task_schema import Candidate, DecisionSchema


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SLOT_FIELDS=('definition','thinking_requirements','role_rules','scope')
POLICY_FIELDS=('presence_policy','requested_unassigned_role_policy','deictic_policy',
               'scope_policy','coexistence_policy','commitment_policy','intent_conditioning',
               'unresolved_boundaries','role_rules')
BOUNDARY_FIELDS=('shared_rule','coexistence_rule','scope_condition','unresolved_constraint','role_rules')


def share_slot_rules(contract):
    """Reference repeated complete strings, never substrings or ontology words.

    The library depends only on the supplied task contract. Original field/list
    placement and rule prose are preserved. Reversibility is not evidence that
    a model follows the referenced rules as well as their literal repetitions.
    """
    strings=[]
    def collect(value):
        if isinstance(value,str):strings.append(value)
        elif isinstance(value,list):
            for item in value:collect(item)
        elif isinstance(value,dict):
            if '$rule' in value:raise ValueError('reserved rule reference key')
            for item in value.values():collect(item)
    collect(contract)
    counts=Counter(strings)
    shared={s:'R'+str(index+1) for index,s in enumerate(dict.fromkeys(
        s for s in strings if len(s)>=64 and counts[s]>1))}
    def replace(value):
        if isinstance(value,str):return {'$rule':shared[value]} if value in shared else value
        if isinstance(value,list):return [replace(v) for v in value]
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        return value
    return dict(
        representation='shared_rules_v1',
        reference_application='Each object {"$rule":"Rk"} means the exact text in rule_texts[Rk] at that field or list position. Apply that text in its local type/policy/boundary context. References are task rules, not answers. The rule applies at every referenced position.',
        rule_texts={rid:s for s,rid in shared.items()},
        contract=replace(contract))


def expand_slot_rules(shared):
    """Strict reconstruction for task-input audits; not a model-side decoder."""
    if shared.get('representation')!='shared_rules_v1':raise ValueError('unknown rule representation')
    library=shared['rule_texts']
    if not isinstance(library,dict) or any(type(k) is not str or type(v) is not str for k,v in library.items()):
        raise ValueError('rule library must map names to strings')
    used=set()
    def expand(value):
        if isinstance(value,list):return [expand(v) for v in value]
        if isinstance(value,dict):
            if '$rule' in value:
                if set(value)!={'$rule'} or type(value['$rule']) is not str or value['$rule'] not in library:
                    raise ValueError('invalid rule reference')
                used.add(value['$rule']);return library[value['$rule']]
            return {k:expand(v) for k,v in value.items()}
        return value
    expanded=expand(shared['contract'])
    if used!=set(library):raise ValueError('unreferenced rule library entry')
    return expanded


def slot_contract(slots, policies, rules, needed, smap):
    """Lossless selected semantic fields; shared identical policy values stored once.

    Audit/evidence/support and historical fields never enter actor knowledge.
    This compiles definitions only; ambiguity masks remain outside actor inputs.
    """
    pmap={p['entity_type']:p for p in policies}
    if len(pmap)!=len(policies) or set(pmap)!={s['entity_type'] for s in slots}:
        raise ValueError('complete unique slot policies required')
    shared={}; selected=[s for s in slots if s['entity_type'] in needed]
    for key in POLICY_FIELDS:
        values=[pmap[s['entity_type']][key] for s in selected]
        if values and all(v==values[0] for v in values):shared[key]=values[0]
    entries={}
    for s in selected:
        body={k:s[k] for k in SLOT_FIELDS}
        body['policy_overrides']={k:pmap[s['entity_type']][k] for k in POLICY_FIELDS if k not in shared}
        entries[smap[s['entity_type']]]=body
    boundaries=[dict(types=[smap[t] for t in r['types']],**{k:r[k] for k in BOUNDARY_FIELDS}) for r in rules]
    return dict(policy_application='Shared policy applies to every listed type. Per-type overrides replace only the named policy fields.',
                shared_policy=shared,types=entries,boundaries=boundaries)


def make_request(audio, panel, intents, slots, rules, size, intent_boundaries=False, policies=None, slot_representation='literal_v1'):
    if slot_representation not in ('literal_v1','shared_rules_v1'):raise ValueError('unknown slot representation')
    if slot_representation!='literal_v1' and policies is None:raise ValueError('shared rules require full policies')
    imap={c['semantic_id']:(string.ascii_uppercase+string.ascii_lowercase+string.digits)[i] for i,c in enumerate(intents)}
    smap={c['entity_type']:'T'+str(i+1) for i,c in enumerate(slots)}
    # Natural words such as time/person/date also happen to be ontology IDs.
    # Never regex-replace them inside prose; neutralize structured keys only.
    rename=lambda s:s
    chosen=panel['families']['slot']['definitions']
    # All incident declared boundaries, with one-hop neighbor definitions. No query/gold routing.
    relevant=[r for r in rules if set(r['types']) & set(chosen)]
    needed=set(chosen)|{t for r in relevant for t in r['types']}
    requirements=(
        'Use the audio and supplied definitions. Fill typed decisions once without generating reasoning or slot values.',
        'Intent is exclusive task membership. Slot presence requires an expressed role, including a field name when its definition permits it; a normally required but unexpressed parameter is not present. Types may coexist. Mention does not establish intent-slot binding.',
        'Intent definitions: '+json.dumps({imap[c['semantic_id']]:
            (dict(definition=c['definition'],includes=c['includes'],excludes=c['excludes']) if intent_boundaries else c['definition'])
            for c in intents},ensure_ascii=False),
        'Slot definitions: '+json.dumps({smap[c['entity_type']]:rename(c['definition']) for c in slots if c['entity_type'] in needed},ensure_ascii=False),
        'Shared boundaries: '+json.dumps([dict(types=[smap[t] for t in r['types']],distinction=rename(r['shared_rule']),coexistence=rename(r['coexistence_rule'])) for r in relevant],ensure_ascii=False))
    representation_info={}
    if policies is not None:
        if not intent_boundaries:raise ValueError('full contract requires full intent boundaries')
        contract=slot_contract(slots,policies,relevant,needed,smap)
        semantic_hash=hashlib.sha256(json.dumps(contract,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if slot_representation=='shared_rules_v1':
            encoded=share_slot_rules(contract)
            if expand_slot_rules(encoded)!=contract:raise ValueError('rule sharing changed contract')
            representation_info['shared_rule_count']=len(encoded['rule_texts'])
            contract=encoded
        representation_info.update(slot_representation=slot_representation,expanded_slot_contract_sha256=semantic_hash)
        requirements=requirements[:3]+('Slot semantic contract: '+json.dumps(contract,ensure_ascii=False),)
    questions=[]
    for i in range(size//2):
        ins=panel['families']['intent']['definitions']
        if i<len(ins):
            questions.append(SpanQuestion('I'+str(i+1),DecisionSchema('I'+str(i+1),'noul','Does this query belong to defined intent '+imap[ins[i]]+'?',allow_unknown=False)))
        if i<len(chosen):
            questions.append(SpanQuestion('S'+str(i+1),DecisionSchema('S'+str(i+1),'noul','Does this query express defined slot type '+smap[chosen[i]]+'?',allow_unknown=False)))
    questions.append(SpanQuestion('Final',DecisionSchema('Final','choice','Which defined intent class describes the spoken request?',
        tuple(Candidate(imap[c['semantic_id']],'Use supplied definition '+imap[c['semantic_id']]+'.') for c in intents),False)))
    return SpanRequest(audio,tuple(questions),'Final',requirements),dict(slot_definitions=len(needed),boundary_rules=len(relevant),
        slot_contract_mode='complete_semantic_fields' if policies is not None else 'historical_short_definitions',**representation_info)


def main():
    import soundfile as sf
    import torch
    from transformers import Qwen3ASRProcessor
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text())
    limit=c['context_limit']
    if type(limit) is not int or limit<1:raise ValueError('explicit positive context limit required')
    if a.output.exists():raise FileExistsError(a.output)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    for path,digest in c['pins'].items():
        if sha(path)!=digest:raise ValueError('changed input '+path)
    read=lambda p:json.loads(Path(p).read_text())
    intents=read(c['intent'])['cards'];slots=read(c['slot'])['types'];rules=read(c['rules'])['rules'];panels=read(c['panels'])['panels']
    policies=None
    if c.get('slot_contract_mode')=='complete_semantic_fields':
        for field in ('intent','slot','rules','policy','panels'):
            if c[field] not in c['pins']:raise ValueError('full contract dependency must be pinned')
        policy=read(c['policy']);rule_doc=read(c['rules'])
        if (policy['source_definition_sha256']!=sha(c['slot']) or
                rule_doc['source_definition_sha256']!=sha(c['slot']) or
                rule_doc['source_policy_sha256']!=sha(c['policy'])):
            raise ValueError('inconsistent full slot contract')
        policies=policy['policies']
    elif c.get('slot_contract_mode') not in (None,'historical_short_definitions'):
        raise ValueError('unknown slot contract mode')
    torch.set_num_threads(4)
    processor=Qwen3ASRProcessor.from_pretrained(c['model'],local_files_only=True)
    a.output.mkdir(parents=True);records=[]
    with (a.output/'records.jsonl').open('w') as stream:
        for asset in c['audio']:
            if sha(asset['audio'])!=asset['sha256']:raise ValueError('audio changed')
            wave,rate=sf.read(asset['audio'],dtype='float32')
            if wave.ndim!=1 or rate!=16000:raise ValueError('mono 16k required')
            for panel in panels:
                common=None
                for size in (0,2,8):
                    request,knowledge=make_request(asset['audio'],panel,intents,slots,rules,size,c['intent_boundaries'],policies,c.get('slot_representation','literal_v1'))
                    knowledge_hash=hashlib.sha256(json.dumps(request.requirements).encode()).hexdigest()
                    if common is not None and common!=knowledge_hash:raise ValueError('knowledge changed across arms')
                    common=knowledge_hash
                    inputs,positions=encode_span_request(processor,request,wave)
                    length=inputs['input_ids'].shape[-1]
                    row=dict(audio_id=asset['id'],audio_selection=asset['selection'],seconds=len(wave)/rate,
                        panel_id=panel['panel_id'],Q_budget=size,actual_auxiliary_questions=len(request.questions)-1,
                        tokens=length,overflow=length>limit,positions=positions.tolist(),knowledge_sha256=knowledge_hash,
                        input_sha256=hashlib.sha256(json.dumps(asdict(request),sort_keys=True).encode()).hexdigest(),**knowledge)
                    stream.write(json.dumps(row)+'\n');stream.flush();records.append(row)
    result=dict(schema='slurp-panel-context-probe-v1',context_limit=limit,intent_boundaries=c['intent_boundaries'],
        slot_contract_mode=c.get('slot_contract_mode','historical_short_definitions'),
        slot_representation=c.get('slot_representation','literal_v1'),
        prose_preserved_verbatim=True,requests=len(records),records_sha256=sha(a.output/'records.jsonl'),
        per_arm={str(size):dict(requests=sum(r['Q_budget']==size for r in records),
            min_tokens=min(r['tokens'] for r in records if r['Q_budget']==size),max_tokens=max(r['tokens'] for r in records if r['Q_budget']==size),
            overflow=sum(r['overflow'] for r in records if r['Q_budget']==size)) for size in (0,2,8)},
        scope='Real CPU processor and unchanged Span alignment for 15 fixed panels on median/longest learning waveforms; not all possible rotating menus, semantic compression equivalence, model parity or training.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
