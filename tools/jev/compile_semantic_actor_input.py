#!/usr/bin/env python3
"""Compile current semantic task inputs for CPU design measurement, not training."""
import json
import string
from pathlib import Path

from build_blind_semantic_annotation import INTENT_FIELDS, project
from build_slurp_annotation_workpack import validate_boundaries
from consolidate_intent_presence_references import catalog, pin, read, require, sha, unique
from probe_slurp_panel_context import share_slot_rules, expand_slot_rules
from share_semantic_blocks import share_blocks, expand_blocks
from multimodal_jev.span_contract import SpanQuestion, SpanRequest
from multimodal_jev.task_schema import Candidate, DecisionSchema

SYMBOLS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def load_design(menu_path, readout_path):
    pins = {}
    for p in (menu_path, readout_path):
        pin(pins, p, sha(p))
    menu, readout = read(menu_path), read(readout_path)
    pin(pins, menu['report'], menu['report_sha256'])
    report = read(menu['report'])
    for p, digest in report['pins'].items():
        pin(pins, p, digest)
    for name, digest in report['files'].items():
        pin(pins, Path(menu['report']).parent/name, digest)
    for p, digest in readout['pins'].items():
        pin(pins, p, digest)
    cards, contracts = catalog(menu['ontology']['intent'], pins)
    policy_text = {}
    for path in read(menu['ontology']['intent'])['policies']:
        d = read(path)
        if '/intent-request-semantics-' in path:
            policy_text['request'] = project(d, ('operator','scope','question','actor_rules','input_contract'))
        elif '/intent-nonrequest-semantics-' in path:
            policy_text['nonrequest'] = project(d, ('greeting','preference','shared'))
        else:
            policy_text[Path(path).stem] = d['rules']
    package = read(menu['ontology']['slot'])
    pin(pins, menu['ontology']['slot'], sha(menu['ontology']['slot']))
    for key in ('definitions','policy','boundaries'):
        pin(pins, package[key], sha(package[key]))
    for key in ('scope_contract','consistency_rule','provider_rule'):
        pin(pins, package[key], package[key+'_sha256'])
    slots = unique(read(package['definitions'])['types'], 'entity_type')
    slot_policy = read(package['policy'])
    require(slot_policy['source_definition_sha256'] == sha(package['definitions']), 'changed slot definition')
    policies = unique(slot_policy['policies'], 'entity_type')
    boundaries = read(package['boundaries'])
    validate_boundaries(boundaries, slots, sha(package['definitions']), sha(package['policy']))
    panels = read(menu['panels'])
    require(set(slots)==set(policies) and len(slots)==55 and len(cards)==65, 'incomplete ontology')
    require([p['panel_id'] for p in panels]==[p['panel_id'] for p in readout['panels']], 'readout panel identity')
    expected = {(q['operator'],q['definition_id']) for p in panels for q in p['process_questions']}
    aliases = {(a['operator'],a['definition_id_offline']):a['definition_alias'] for a in readout['semantic_aliases']}
    require(set(aliases)==expected and len(set(aliases.values()))==len(aliases), 'semantic alias identity')
    for p, layout in zip(panels,readout['panels'],strict=True):
        require([(q['operator'],q['definition_id']) for q in p['process_questions']] ==
                [(q['operator'],q['definition_id_offline']) for q in layout['semantic_questions']], 'question binding changed')
        for index,q in enumerate(layout['semantic_questions']):
            require(q['definition_alias']==aliases[q['operator'],q['definition_id_offline']] and
                    q['span_position']==index and q['question_alias']==f'Q{index:02d}' and
                    q['boolean_output_symbols']=={'A':False,'B':True}, 'semantic readout mismatch')
    return dict(native=unique(read(menu['native_menu']),'native_id'), readout=readout,
                cards=cards, acts={k:v['act_policy'] for k,v in contracts.items()},
                intent_policies=policy_text, slots=slots, slot_policies=policies,
                boundaries=boundaries['rules'], field_groups=package['runtime_field_groups'],
                aliases=aliases, pins=pins)


def semantic_knowledge(design, panel, representation):
    require(representation in ('literal_v1','shared_rules_v1','shared_blocks_v1'), 'unknown representation')
    qs = panel['semantic_questions']
    intents = [q['definition_id_offline'] for q in qs if q['operator']=='I_present']
    selected = {q['definition_id_offline'] for q in qs if q['operator']=='S_present'}
    incident = [b for b in design['boundaries'] if selected & set(b['types'])]
    needed = selected | {t for b in incident for t in b['types']}
    alias = lambda family,key:design['aliases'][family,key]
    fields = design['field_groups']
    # Prose remains verbatim. Only structured identity keys use readout aliases.
    contract = dict(
        intent=dict(definitions={alias('I_present',k):{**project(design['cards'][k],INTENT_FIELDS),
                            'act_policy':design['acts'][k]} for k in intents},
                    policies=design['intent_policies']),
        slot=dict(definitions={alias('S_present',k):project(design['slots'][k],fields['type']) for k in sorted(needed)},
                  policies={alias('S_present',k):project(design['slot_policies'][k],fields['policy']) for k in sorted(needed)},
                  boundaries=[dict(types=[alias('S_present',t) for t in b['types']],
                                   **project(b,fields['boundary'])) for b in incident]))
    for key in needed:
        require(all(k in design['slots'][key] for k in fields['type']) and
                all(k in design['slot_policies'][key] for k in fields['policy']), 'missing slot field')
    require(all(all(k in b for k in fields['boundary']) for b in incident), 'missing boundary field')
    encoded = contract if representation=='literal_v1' else share_slot_rules(contract)
    if representation=='shared_rules_v1':
        require(expand_slot_rules(encoded)==contract, 'rule-sharing content changed')
    elif representation=='shared_blocks_v1':
        encoded = share_blocks(encoded)
        require(expand_blocks(encoded)==contract, 'block-sharing content changed')
    return encoded, contract, dict(selected_intents=len(intents),selected_slots=len(selected),
        supplied_slot_definitions=len(needed),incident_boundaries=len(incident))


def make_request(design, audio, panel_id, arm, native_layout='canonical', representation='shared_rules_v1'):
    require(arm in ('Direct','Q_input','Q_credit'), 'unknown arm')
    panel = next((p for p in design['readout']['panels'] if p['panel_id']==panel_id),None)
    require(panel is not None, 'unknown panel')
    require(native_layout in design['readout']['native_layouts'], 'unknown native layout')
    entries = design['readout']['native_layouts'][native_layout]['entries']
    require({e['native_id'] for e in entries}==set(design['native']) and len(entries)==len(design['native']), 'native support mismatch')
    require({e['output_symbol'] for e in entries}==set(SYMBOLS[:len(entries)]) and
            len({e['candidate_alias'] for e in entries})==len(entries), 'native symbols or aliases invalid')
    encoded, expanded, info = semantic_knowledge(design,panel,representation)
    native = [dict(candidate_alias=e['candidate_alias'],output_symbol=e['output_symbol'],
                   descriptor=design['native'][e['native_id']]['descriptor'])
              for e in sorted(entries,key=lambda e:e['display_position'])]
    requirements = (
        'Use only the audio and supplied task definitions. Fill all configured typed Spans in one forward pass; do not generate reasoning or slot values.',
        'Native classes and semantic predicates are separate namespaces. Native category descriptors specify the classification menu, not necessary-and-sufficient semantic definitions. Semantic predicates may overlap; do not force one-hot, prune native candidates, or infer OOD from all-false predicates.',
        'I_present asks whether an expressed communicative act satisfies its definition and act policy. S_present asks whether an expressed semantic role satisfies its type and presence policy across the complete communicative content, not execution state or intent-slot binding. Q IDs identify local readouts; I/S aliases identify supplied definitions, including boundary neighbors. A/B encode false/true locally at each semantic Span.',
        'Native classification menu: '+json.dumps(native,ensure_ascii=False),
        'Semantic definitions and policies: '+json.dumps(encoded,ensure_ascii=False))
    questions=[]
    if arm!='Direct':
        for q in panel['semantic_questions']:
            text = ('Does this query express an intention satisfying supplied definition ' if q['operator']=='I_present'
                    else 'Does this query express a semantic role satisfying supplied type ')
            questions.append(SpanQuestion(q['question_alias'],DecisionSchema(q['question_alias'],'noul',text+q['definition_alias']+' under its policy?',allow_unknown=False)))
    # Native display order is independent of readout symbol order. span_labels
    # binds A..7 by candidate sequence, so arrange this sequence by that symbol.
    ordered = sorted(entries,key=lambda e:SYMBOLS.index(e['output_symbol']))
    candidates=tuple(Candidate(e['candidate_alias'],'Use native classification descriptor '+e['candidate_alias']+'.') for e in ordered)
    questions.append(SpanQuestion('Final',DecisionSchema('Final','choice','Which native category describes the audio under the supplied classification menu?',candidates,False)))
    require(len(questions)-1==panel['arms'][arm]['final_span_position'],'Final position mismatch')
    return SpanRequest(audio,tuple(questions),'Final',requirements), expanded, info
